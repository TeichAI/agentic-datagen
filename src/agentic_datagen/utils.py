"""Utility functions for loading prompts."""

import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _stringify_content(value: Any) -> str | None:
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None

    if isinstance(value, (int, float)):
        return str(value)

    if isinstance(value, Iterable):
        parts: List[str] = []
        for item in value:
            if isinstance(item, str):
                candidate = item.strip()
                if candidate:
                    parts.append(candidate)
            elif isinstance(item, dict):
                nested = _stringify_content(item.get("text"))
                if nested:
                    parts.append(nested)
        if parts:
            return "\n".join(parts)
    return None


def _extract_prompts_from_json_record(record: Any) -> List[str]:
    prompts: List[str] = []
    if isinstance(record, dict):
        messages = record.get("messages")
        if isinstance(messages, list):
            for message in messages:
                if not isinstance(message, dict):
                    continue
                role = str(message.get("role", "")).lower()
                if role and role != "user":
                    continue
                content = _stringify_content(message.get("content"))
                if content:
                    prompts.append(content)

        for key in ("prompt", "input", "question", "task", "query"):
            if key in record:
                content = _stringify_content(record[key])
                if content:
                    prompts.append(content)

    return prompts


def _extract_prompts_from_json_payload(payload: Any) -> List[str]:
    if isinstance(payload, list):
        prompts: List[str] = []
        for item in payload:
            prompts.extend(_extract_prompts_from_json_record(item))
        return prompts

    return _extract_prompts_from_json_record(payload)


def _prompt_sort_key(path: Path) -> tuple[int, str]:
    stem = path.stem
    try:
        return (0, str(int(stem)))
    except ValueError:
        return (1, stem)


def _load_markdown_prompts(directory: Path) -> List[str]:
    prompts: List[str] = []
    for prompt_file in sorted(directory.glob("*.md"), key=_prompt_sort_key):
        text = prompt_file.read_text(encoding="utf-8").strip()
        if text:
            prompts.append(text)
    return prompts


def _load_text_prompts(path: Path) -> List[str]:
    prompts: List[str] = []
    with path.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            normalized = raw_line.strip()
            if not normalized:
                continue
            prompts.append(normalized)
    return prompts


ARTIFACT_PROMPT_KEYWORDS = (
    "build",
    "create",
    "make",
    "develop",
    "website",
    "landing page",
    "dashboard",
    "portfolio",
    "site",
    "app",
    "tool",
    "page",
)

MUTATING_TOOL_NAMES = {
    "write_file",
    "edit_file",
    "run_command",
    "write",
    "edit",
    "bash",
}

EMPTY_WORKSPACE_TERMINAL_ERRORS = {
    "Session reported completion but workspace has no preserved files",
    "Session did not produce any workspace files",
}


def prompt_requires_artifact(prompt: str) -> bool:
    lowered = prompt.lower()
    return any(keyword in lowered for keyword in ARTIFACT_PROMPT_KEYWORDS)


def count_workspace_files(workspace_dir: Path) -> int:
    if not workspace_dir.exists():
        return 0
    return sum(1 for item in workspace_dir.rglob("*") if item.is_file())


def _tool_result_succeeded(result: Any) -> bool:
    if not isinstance(result, dict):
        return result is not None
    if result.get("success") is False:
        return False
    if result.get("isError") is True or result.get("is_error") is True:
        return False
    if result.get("error") not in (None, ""):
        return False
    if result.get("success") is True:
        return True
    status = result.get("status")
    if isinstance(status, str):
        normalized = status.strip().lower()
        if normalized in {"completed", "success", "succeeded", "ok", "done"}:
            return True
        if normalized in {"error", "failed", "failure", "cancelled", "canceled", "timeout", "timed_out"}:
            return False
    return bool(result.get("result") is not None or result.get("output"))


def count_successful_mutating_tool_calls(tool_calls: Any) -> int:
    if not isinstance(tool_calls, list):
        return 0
    count = 0
    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            continue
        tool_name = tool_call.get("tool")
        if tool_name not in MUTATING_TOOL_NAMES:
            continue
        if _tool_result_succeeded(tool_call.get("result")):
            count += 1
    return count


def apply_workspace_completion_guardrails(
    session_data: Dict[str, Any],
    *,
    prompt: str,
    workspace_dir: Path,
) -> Dict[str, Any]:
    normalized = dict(session_data)
    workspace_file_count = count_workspace_files(workspace_dir)
    normalized["workspace_file_count"] = workspace_file_count
    normalized["workspace_has_artifacts"] = workspace_file_count > 0
    normalized["successful_mutating_tool_calls"] = count_successful_mutating_tool_calls(
        normalized.get("tool_calls")
    )
    if (
        normalized.get("completed") is True
        and not normalized.get("error")
        and prompt_requires_artifact(prompt)
        and workspace_file_count == 0
    ):
        normalized["completed"] = False
        normalized["retryable"] = False
        if normalized["successful_mutating_tool_calls"] > 0:
            normalized["error"] = "Session reported completion but workspace has no preserved files"
        else:
            normalized["error"] = "Session did not produce any workspace files"
    return normalized


def load_guarded_session_state(
    state_file: Path,
    *,
    prompt: str,
    workspace_dir: Path,
) -> Optional[Dict[str, Any]]:
    if not state_file.exists():
        return None
    try:
        with state_file.open("r", encoding="utf-8") as f:
            state = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if state.get("prompt") != prompt:
        return None
    normalized = apply_workspace_completion_guardrails(
        state,
        prompt=prompt,
        workspace_dir=workspace_dir,
    )
    if (
        normalized.get("workspace_file_count") == 0
        and normalized.get("error") in EMPTY_WORKSPACE_TERMINAL_ERRORS
    ):
        delete_session_state(workspace_dir)
        return None
    return normalized


def get_session_state_path(workspace_dir: Path) -> Path:
    digest = hashlib.sha256(str(workspace_dir.resolve()).encode("utf-8")).hexdigest()
    return Path(tempfile.gettempdir()) / "agentic-datagen-session-state" / f"{digest}.json"


def migrate_legacy_session_state(workspace_dir: Path, state_file: Path) -> None:
    legacy_state_file = workspace_dir / "session_state.json"
    if state_file.exists() or not legacy_state_file.exists():
        return
    state_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.move(str(legacy_state_file), str(state_file))
        return
    except OSError:
        pass
    try:
        state_file.write_text(legacy_state_file.read_text(encoding="utf-8"), encoding="utf-8")
        legacy_state_file.unlink()
    except OSError:
        return


def delete_session_state(workspace_dir: Path) -> None:
    state_file = get_session_state_path(workspace_dir)
    try:
        state_file.unlink()
    except FileNotFoundError:
        return
    except OSError:
        return
    try:
        if state_file.parent.exists() and not any(state_file.parent.iterdir()):
            state_file.parent.rmdir()
    except OSError:
        return


def load_prompts(path: Path) -> List[str]:
    """Load prompts from various sources."""
    if not path.exists():
        raise FileNotFoundError(f"Prompt source not found: {path}")

    if path.is_dir():
        return _load_markdown_prompts(path)

    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".json"}:
        prompts: List[str] = []

        if suffix == ".jsonl":
            with path.open("r", encoding="utf-8") as fh:
                for raw_line in fh:
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"Invalid JSONL line in {path}: {exc}"
                        ) from exc
                    prompts.extend(_extract_prompts_from_json_payload(payload))
        else:
            text = path.read_text(encoding="utf-8")
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
            prompts.extend(_extract_prompts_from_json_payload(payload))

        normalized_prompts: List[str] = []
        for prompt in prompts:
            normalized = prompt.strip()
            if not normalized:
                continue
            normalized_prompts.append(normalized)
        return normalized_prompts

    if suffix == ".md":
        text = path.read_text(encoding="utf-8").strip()
        return [text] if text else []

    if suffix == ".txt":
        return _load_text_prompts(path)

    raise ValueError(f"Unsupported prompt source type: {suffix}")

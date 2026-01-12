"""Utility functions for loading prompts."""

import json
from pathlib import Path
from typing import Any, Iterable, List


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

        for key in ("prompt", "input", "question", "task"):
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
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            normalized = raw_line.strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            prompts.append(normalized)
    return prompts


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

        seen = set()
        unique_prompts: List[str] = []
        for prompt in prompts:
            normalized = prompt.strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            unique_prompts.append(normalized)
        return unique_prompts

    if suffix == ".md":
        text = path.read_text(encoding="utf-8").strip()
        return [text] if text else []

    if suffix == ".txt":
        return _load_text_prompts(path)

    raise ValueError(f"Unsupported prompt source type: {suffix}")

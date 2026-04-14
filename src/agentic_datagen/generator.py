import json
import logging
import os
import shutil
import sys
import threading
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml

from .formatter import Formatter
from .run_manifest import RunManifest
from .session_engines import create_session_engine, get_engine_tool_definitions
from .tool_registry import ToolRegistry
from .utils import delete_session_state


DEFAULT_ENABLED_TOOLS = [
    "read_file",
    "write_file",
    "edit_file",
    "list_directory",
    "search_code",
    "run_command",
    "web_search",
    "workspace_snapshot",
    "context7:*",
]

PROVIDER_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1/chat/completions",
        "api_key_env": "OPENROUTER_API_KEY",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1/chat/completions",
        "api_key_env": "OPENAI_API_KEY",
    },
}


DEFAULT_CONFIG: Dict[str, Any] = {
    "api": {
        "provider": "openrouter",
        "base_url": "https://openrouter.ai/api/v1/chat/completions",
        "model": None,
        "api_key": None,
        "api_key_env": "OPENROUTER_API_KEY",
        "reasoning_effort": None,
        "max_retries": 5,
        "backoff_base_seconds": 2.0,
        "backoff_max_seconds": 60.0,
        "timeout": 120,
        "searxng_url": None,
    },
    "prompts": {
        "source": "prompts.txt",
        "limit": None,
        "shuffle": False,
    },
    "workspace": {
        "base_dir": "sandbox",
        "cleanup": True,
        "preserve_on_error": True,
        "command_runner": {
            "mode": "docker",
            "tool_scope": "all",
            "docker_image": "agentic-datagen-session-runtime:latest",
        },
    },
    "agent": {
        "engine": "native",
        "max_turns": 50,
        "system_prompt": None,
        "tools_enabled": list(DEFAULT_ENABLED_TOOLS),
    },
    "tools": {
        "custom_python_modules": [],
        "strict_mcp": False,
        "mcp_servers": {},
    },
    "output": {
        "dataset_file": "datasets/agentic_dataset.jsonl",
        "append_mode": True,
        "sanitize_existing_dataset": True,
    },
    "processing": {
        "concurrency": 1,
        "resume": True,
        "retryable_session_max_attempts": 2,
    },
    "logging": {
        "level": "INFO",
        "console": True,
    },
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(existing, value)
        else:
            merged[key] = value
    return merged


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _normalize_base_url(provider: str, base_url: Any) -> Any:
    if not isinstance(base_url, str):
        return base_url
    normalized = base_url.strip()
    if not normalized:
        return None
    parsed = urlparse(normalized)
    path = (parsed.path or "").rstrip("/")
    if path.endswith("/chat/completions") or path == "/chat/completions":
        return normalized.rstrip("/")
    if path.endswith("/v1") or path == "/v1":
        return f"{normalized.rstrip('/')}/chat/completions"
    if provider == "openai" and path in {"", "/"}:
        return f"{normalized.rstrip('/')}/v1/chat/completions"
    return normalized.rstrip("/")


def _is_local_base_url(base_url: Any) -> bool:
    if not isinstance(base_url, str) or not base_url.strip():
        return False
    parsed = urlparse(base_url)
    hostname = (parsed.hostname or "").lower()
    return hostname in {"localhost", "127.0.0.1", "0.0.0.0", "::1"}


def normalize_config(raw_config: Dict[str, Any]) -> Dict[str, Any]:
    raw = raw_config if isinstance(raw_config, dict) else {}
    config = _deep_merge(DEFAULT_CONFIG, raw)

    model_config = _as_dict(raw.get("model"))
    tools_config = _as_dict(raw.get("tools"))
    tool_web_search_config = _as_dict(tools_config.get("web_search"))
    api_config = config.setdefault("api", {})
    provider = str(api_config.get("provider") or "openrouter").strip().lower() or "openrouter"
    provider_defaults = PROVIDER_DEFAULTS.get(provider, {})
    api_config["provider"] = provider
    if not api_config.get("base_url"):
        api_config["base_url"] = provider_defaults.get("base_url")
    if not api_config.get("api_key_env"):
        api_config["api_key_env"] = provider_defaults.get("api_key_env")

    if model_config:
        agent_config = config.setdefault("agent", {})
        model_provider = model_config.get("provider", api_config.get("provider"))
        provider = str(model_provider or "openrouter").strip().lower() or "openrouter"
        provider_defaults = PROVIDER_DEFAULTS.get(provider, {})
        api_config["provider"] = provider
        if "base_url" in model_config:
            api_config["base_url"] = model_config.get("base_url")
        elif raw.get("api", {}).get("base_url") in (None, ""):
            api_config["base_url"] = provider_defaults.get("base_url", api_config.get("base_url"))
        model_name = model_config.get("name") or model_config.get("model")
        if model_name is not None:
            api_config["model"] = model_name
        for key in [
            "api_key",
            "api_key_env",
            "reasoning_effort",
            "max_retries",
            "backoff_base_seconds",
            "backoff_max_seconds",
            "timeout",
        ]:
            if key in model_config:
                api_config[key] = model_config.get(key)
        if "api_key_env" not in model_config and raw.get("api", {}).get("api_key_env") in (None, ""):
            api_config["api_key_env"] = provider_defaults.get("api_key_env", api_config.get("api_key_env"))
        if "system_prompt" in model_config:
            agent_config["system_prompt"] = model_config.get("system_prompt")
        if "max_turns" in model_config:
            agent_config["max_turns"] = model_config.get("max_turns")

    api_config["base_url"] = _normalize_base_url(
        str(api_config.get("provider") or "openrouter").strip().lower() or "openrouter",
        api_config.get("base_url"),
    )

    enabled_tools = tools_config.get("enabled")
    if isinstance(enabled_tools, list):
        config.setdefault("agent", {})["tools_enabled"] = enabled_tools

    if "custom_python_modules" in tools_config:
        config.setdefault("tools", {})["custom_python_modules"] = tools_config.get(
            "custom_python_modules"
        ) or []
    if "strict_mcp" in tools_config:
        config.setdefault("tools", {})["strict_mcp"] = bool(tools_config.get("strict_mcp"))
    if "mcp_servers" in tools_config and isinstance(tools_config.get("mcp_servers"), (dict, list)):
        config.setdefault("tools", {})["mcp_servers"] = tools_config.get("mcp_servers")

    searxng_url = tool_web_search_config.get("searxng_url") or tools_config.get("searxng_url")
    if searxng_url is not None:
        config.setdefault("api", {})["searxng_url"] = searxng_url

    return config


def _derive_dataset_title(dataset_file: Path) -> str:
    stem = dataset_file.stem.strip()
    if not stem:
        return "Agentic Dataset"
    return " ".join(part.capitalize() for part in stem.replace("_", " ").replace("-", " ").split())


def _default_dataset_description(model: str | None) -> str:
    if model:
        return f"This is an agentic coding dataset generated using {model}."
    return "This is an agentic coding dataset generated by agentic-datagen."


def _default_pretty_name(model: str | None) -> str:
    return f"{model or 'agentic-datagen'} coding agent traces"


def _string_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    items: List[str] = []
    for item in value:
        text = str(item).strip() if item is not None else ""
        if text:
            items.append(text)
    return items


def _merge_unique_lists(*values: List[str]) -> List[str]:
    merged: List[str] = []
    seen: Set[str] = set()
    for group in values:
        for item in group:
            if item not in seen:
                seen.add(item)
                merged.append(item)
    return merged


def _example_argument_value(schema: Dict[str, Any]) -> Any:
    if not isinstance(schema, dict):
        return "..."
    enum_values = schema.get("enum")
    if isinstance(enum_values, list) and enum_values:
        return enum_values[0]
    arg_type = schema.get("type")
    if arg_type == "integer":
        return 1
    if arg_type == "number":
        return 1
    if arg_type == "boolean":
        return False
    if arg_type == "array":
        return []
    if arg_type == "object":
        return {}
    return "..."


def _example_tool_payload(tool_definitions: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    fallback_tool = {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create or overwrite a file in the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["file_path", "content"],
            },
        },
    }
    tool_definition = fallback_tool
    for candidate in tool_definitions:
        if isinstance(candidate, dict) and isinstance(candidate.get("function"), dict):
            tool_definition = candidate
            break

    function = tool_definition.get("function", {})
    parameters = function.get("parameters", {}) if isinstance(function, dict) else {}
    properties = parameters.get("properties", {}) if isinstance(parameters, dict) else {}
    required = parameters.get("required", []) if isinstance(parameters, dict) else []
    arguments: Dict[str, Any] = {}
    if isinstance(properties, dict):
        for name in required:
            if isinstance(name, str) and name in properties:
                arguments[name] = _example_argument_value(properties.get(name) or {})

    return tool_definition, {
        "id": "call_1",
        "type": "function",
        "index": 0,
        "function": {
            "name": function.get("name", "write_file") if isinstance(function, dict) else "write_file",
            "arguments": arguments,
        },
    }


def build_dataset_readme(
    config: Dict[str, Any],
    dataset_file: Path,
    row_count: int,
    tool_definitions: List[Dict[str, Any]],
) -> str:
    output_config = config.get("output", {}) or {}
    card_config = output_config.get("dataset_card", {}) or {}
    api_config = config.get("api", {}) or {}
    agent_config = config.get("agent", {}) or {}

    title = card_config.get("title") or _derive_dataset_title(dataset_file)
    description = card_config.get("description") or _default_dataset_description(
        api_config.get("model")
    )
    license_name = card_config.get("license") or "apache-2.0"
    config_name = card_config.get("config_name") or "default"
    split_name = card_config.get("split") or "train"
    system_prompt = agent_config.get("system_prompt") or "You are a coding agent."
    tool_names = [
        tool.get("function", {}).get("name")
        for tool in tool_definitions
        if isinstance(tool, dict) and isinstance(tool.get("function"), dict)
    ]
    tool_names = [name for name in tool_names if isinstance(name, str) and name]
    tool_line = ", ".join(f"`{name}`" for name in tool_names) if tool_names else "No tools configured"
    model = api_config.get("model")
    reasoning_effort = api_config.get("reasoning_effort") or "not set"
    pretty_name = card_config.get("pretty_name") or _default_pretty_name(model)
    task_categories = _merge_unique_lists(
        ["text-generation"],
        _string_list(card_config.get("task_categories")),
    )
    tags = _merge_unique_lists(
        ["agent-traces", "coding-agent"],
        _string_list(card_config.get("tags")),
    )
    example_tool_definition, example_tool_call = _example_tool_payload(tool_definitions)
    base_metadata = {
        "prompt_id": "prompt_000000_abc123",
        "run_id": "run_20260414T000000Z",
        "session_id": "session_000000",
        "turns": 1,
        "completed": True,
        "tool_calls_count": 0,
        "error": None,
        "retryable": False,
    }
    base_usage = {
        "prompt_tokens": 256,
        "completion_tokens": 128,
        "reasoning_tokens": 32,
        "total_tokens": 416,
        "cost": 0.0,
    }
    non_tool_row = {
        "prompt": "...",
        "tools": tool_definitions,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "..."},
            {
                "role": "assistant",
                "reasoning_content": "...",
                "content": "Final answer...",
            },
        ],
        "metadata": base_metadata,
        "usage": base_usage,
    }
    tool_row = {
        "prompt": "...",
        "tools": [example_tool_definition],
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "..."},
            {
                "role": "assistant",
                "reasoning_content": "...",
                "content": "",
                "tool_calls": [example_tool_call],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "name": example_tool_call["function"]["name"],
                "content": "...",
            },
            {
                "role": "assistant",
                "content": "Final answer after tool result...",
            },
        ],
        "metadata": {
            **base_metadata,
            "turns": 2,
            "tool_calls_count": 1,
        },
        "usage": {
            **base_usage,
            "prompt_tokens": 512,
            "completion_tokens": 160,
            "reasoning_tokens": 48,
            "total_tokens": 720,
        },
    }

    return "\n".join(
        [
            "---",
            f"pretty_name: {json.dumps(pretty_name)}",
            "task_categories:",
            *[f"- {json.dumps(category)}" for category in task_categories],
            "tags:",
            *[f"- {json.dumps(tag)}" for tag in tags],
            f"license: {license_name}",
            "configs:",
            f"- config_name: {config_name}",
            "  data_files:",
            f"  - split: {json.dumps(split_name)}",
            f"    path: {json.dumps(dataset_file.name)}",
            "---",
            "",
            f"# {title}",
            "",
            description,
            "",
            f"This dataset currently contains {row_count} rows.",
            "",
            "## Formatting guide",
            "",
            "Each row in the dataset includes the same top-level keys: `prompt`, `tools`, `messages`, `metadata`, and `usage`.",
            "",
            "### Completed row without tool use",
            "",
            "```json",
            json.dumps(non_tool_row, ensure_ascii=False, indent=2),
            "```",
            "",
            "### Completed row with tool use",
            "",
            "```json",
            json.dumps(tool_row, ensure_ascii=False, indent=2),
            "```",
            "",
            "`reasoning_content` is only included when a reasoning trace is present. Assistant `tool_calls` stay inside the assistant message, tool responses stay in `tool` messages, and the top-level `tools`, `metadata`, and `usage` fields are preserved on every row.",
            "",
            "## Generation metadata",
            "",
            f"- Model: `{model}`" if model else "- Model: not set",
            f"- Reasoning effort: `{reasoning_effort}`",
            f"- Enabled tools: {tool_line}",
            "- Row metadata fields: `prompt_id`, `run_id`, `session_id`, `turns`, `completed`, `tool_calls_count`, `error`, `retryable`.",
            "- Usage fields: `prompt_tokens`, `completion_tokens`, `reasoning_tokens`, `total_tokens`, `cost`.",
        ]
    ) + "\n"


class AgenticDatasetGenerator:
    """Main orchestrator for agentic dataset generation."""

    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        self.formatter = Formatter()
        self.write_lock = threading.Lock()
        self.run_id = self.config.get("output", {}).get("run_id") or datetime.now(
            timezone.utc
        ).strftime("run_%Y%m%dT%H%M%SZ")

        self.api_key = self._get_api_key()
        self.config["api"]["api_key"] = self.api_key

        self.base_workspace_dir = Path(self.config["workspace"]["base_dir"])
        self.base_workspace_dir.mkdir(parents=True, exist_ok=True)

        self.output_file = Path(self.config["output"]["dataset_file"])
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        dataset_readme_path = self.config.get("output", {}).get("dataset_readme_file")
        if dataset_readme_path:
            self.dataset_readme_file = Path(dataset_readme_path)
        else:
            self.dataset_readme_file = self.output_file.with_name("DATASET_README.md")
        self.dataset_readme_file.parent.mkdir(parents=True, exist_ok=True)
        manifest_path = self.config.get("output", {}).get("run_manifest_file")
        if manifest_path:
            self.run_manifest_file = Path(manifest_path)
        else:
            self.run_manifest_file = self.output_file.with_suffix(".manifest.json")
        self.run_manifest_file.parent.mkdir(parents=True, exist_ok=True)
        self.run_manifest = RunManifest(self.run_manifest_file, run_id=self.run_id)

        if not self.config.get("output", {}).get("append_mode", True):
            if self.output_file.exists():
                self.output_file.unlink()

        self.error_output_file = None

        if self.config.get("output", {}).get("sanitize_existing_dataset", True):
            self._sanitize_output_dataset()

        self.enabled_tools = self.config["agent"].get("tools_enabled", [])
        self.tool_definitions = get_engine_tool_definitions(
            self.config,
            self.enabled_tools,
        )

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(config_path, "r", encoding="utf-8") as f:
            raw_config = yaml.safe_load(f) or {}
        return normalize_config(raw_config)

    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        log_config = self.config.get("logging", {})
        level = getattr(logging, log_config.get("level", "INFO"))

        logger = logging.getLogger("agentic_datagen")
        logger.setLevel(level)

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        if log_config.get("console", True):
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        if "log_file" in log_config:
            file_handler = logging.FileHandler(log_config["log_file"])
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        return logger

    def _get_api_key(self) -> str:
        """Get API key from config or environment."""
        api_config = self.config["api"]
        provider = str(api_config.get("provider") or "").strip().lower()
        base_url = api_config.get("base_url")

        if "api_key" in api_config and api_config["api_key"]:
            api_key = api_config["api_key"]
            self.logger.info(
                f"Using API Key from config: {api_key[:4]}...{api_key[-4:]}"
            )
            return api_key

        env_var = api_config.get("api_key_env", "OPENROUTER_API_KEY")
        api_key = os.getenv(env_var)

        if not api_key:
            from dotenv import dotenv_values

            api_key = dotenv_values(".env").get(env_var)

        if not api_key:
            old_env = Path("old/.env")
            if old_env.exists():
                from dotenv import dotenv_values

                api_key = dotenv_values(old_env).get(env_var)

        if not api_key and provider == "openai" and _is_local_base_url(base_url):
            self.logger.info("No API key configured for local OpenAI-compatible endpoint; continuing without auth")
            return ""

        if not api_key:
            raise ValueError(
                f"Missing API key. Provide 'api_key' in config or set {env_var} environment variable."
            )

        self.logger.info(
            f"Using API Key from {env_var}: {api_key[:4]}...{api_key[-4:]}"
        )
        return api_key

    def _load_prompts(self) -> List[str]:
        """Load prompts from configured source."""
        from .utils import load_prompts

        prompts_config = self.config["prompts"]
        source_path = Path(prompts_config["source"])
        prompts = load_prompts(source_path)

        if prompts_config.get("shuffle", False):
            import random

            random.shuffle(prompts)

        limit = prompts_config.get("limit")
        if limit and limit > 0:
            prompts = prompts[:limit]

        return prompts

    def _load_completed_prompts(self) -> Tuple[Set[str], Counter[str]]:
        """Load completed prompt IDs plus prompt-text counts for resume fallback."""
        completed_ids: Set[str] = set()
        completed_prompt_counts: Counter[str] = Counter()

        if not self.output_file.exists():
            return completed_ids, completed_prompt_counts

        with self.output_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if not self.formatter.is_training_safe_entry(entry):
                    continue

                metadata = entry.get("metadata") or {}
                prompt_id = metadata.get("prompt_id") if isinstance(metadata, dict) else None
                prompt = entry.get("prompt")
                if isinstance(prompt, str) and prompt.strip():
                    completed_prompt_counts[prompt.strip()] += 1
                if isinstance(prompt_id, str) and prompt_id.strip():
                    completed_ids.add(prompt_id.strip())

        return completed_ids, completed_prompt_counts

    def _sanitize_output_dataset(self) -> None:
        if not self.output_file.exists():
            return

        temp_file = self.output_file.with_name(f"{self.output_file.name}.tmp")
        valid_entries = 0
        removed_entries = 0
        rewritten_entries = 0

        with self.output_file.open("r", encoding="utf-8") as src, temp_file.open(
            "w", encoding="utf-8"
        ) as dst:
            for line in src:
                stripped = line.strip()
                if not stripped:
                    continue

                try:
                    entry = json.loads(stripped)
                except json.JSONDecodeError:
                    removed_entries += 1
                    continue

                if not self.formatter.is_training_safe_entry(entry):
                    removed_entries += 1
                    continue

                canonical_line = self.formatter.to_jsonl_line(entry)
                if canonical_line != stripped:
                    rewritten_entries += 1
                dst.write(canonical_line + "\n")
                valid_entries += 1

        if removed_entries or rewritten_entries:
            os.replace(temp_file, self.output_file)
            if removed_entries:
                self.logger.info(
                    "Sanitized existing dataset by keeping %s valid rows and removing %s invalid rows",
                    valid_entries,
                    removed_entries,
                )
            else:
                self.logger.info(
                    "Rewrote %s existing rows in %s to canonical dataset format",
                    rewritten_entries,
                    self.output_file,
                )
            return

        if temp_file.exists():
            temp_file.unlink()

    def _validate_runtime_prerequisites(self) -> None:
        probe_workspace = self.base_workspace_dir / ".runtime_probe"
        probe_workspace.mkdir(parents=True, exist_ok=True)
        registry = ToolRegistry(probe_workspace, config=self.config)
        try:
            registry.validate_runtime_prerequisites()
        finally:
            registry.close()
            try:
                probe_workspace.rmdir()
            except OSError:
                pass

    def _count_output_rows(self) -> int:
        if not self.output_file.exists():
            return 0
        with self.output_file.open("r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())

    def _write_dataset_readme(self) -> None:
        readme = build_dataset_readme(
            self.config,
            self.output_file,
            self._count_output_rows(),
            self.tool_definitions,
        )
        self.dataset_readme_file.write_text(readme, encoding="utf-8")
        self.logger.info("Dataset README written to %s", self.dataset_readme_file)

    def _create_workspace(self, session_id: str) -> Path:
        """Create a workspace directory for a session."""
        workspace_dir = self.base_workspace_dir / session_id
        workspace_dir.mkdir(parents=True, exist_ok=True)
        return workspace_dir

    def _cleanup_workspace(self, workspace_dir: Path):
        """Remove workspace directory with retry for stubborn files."""
        if not workspace_dir.exists():
            delete_session_state(workspace_dir)
            return
        for attempt in range(3):
            try:
                shutil.rmtree(workspace_dir)
                delete_session_state(workspace_dir)
                return
            except OSError as e:
                if attempt < 2:
                    import time

                    time.sleep(0.5 * (attempt + 1))
                else:
                    import subprocess

                    try:
                        subprocess.run(
                            ["rm", "-rf", str(workspace_dir)],
                            timeout=30,
                            check=False,
                        )
                        delete_session_state(workspace_dir)
                    except Exception:
                        self.logger.warning(f"Could not fully clean {workspace_dir}: {e}")

    def _retryable_session_max_attempts(self) -> int:
        processing_config = self.config.get("processing", {}) or {}
        try:
            return max(1, int(processing_config.get("retryable_session_max_attempts", 2)))
        except (TypeError, ValueError):
            return 2

    def _session_retry_delay(self, attempt: int) -> float:
        api_config = self.config.get("api", {}) or {}
        try:
            base_delay = max(0.0, float(api_config.get("backoff_base_seconds", 2.0) or 0.0))
        except (TypeError, ValueError):
            base_delay = 2.0
        try:
            max_delay = max(0.0, float(api_config.get("backoff_max_seconds", 60.0) or 0.0))
        except (TypeError, ValueError):
            max_delay = 60.0
        delay = base_delay * (2 ** max(0, attempt - 1))
        return min(delay, max_delay) if max_delay > 0 else delay

    def _process_prompt(self, prompt: str, index: int) -> Optional[Dict[str, Any]]:
        """Process a single prompt and return formatted entry."""
        prompt_id = self.run_manifest.make_prompt_id(index, prompt)
        session_id = f"session_{index:06d}"
        workspace_dir = self._create_workspace(session_id)
        preserve_on_error = self.config["workspace"].get("preserve_on_error", True)
        max_attempts = self._retryable_session_max_attempts()

        self.logger.info(f"Processing prompt {index}: {prompt[:80]}...")
        for attempt in range(1, max_attempts + 1):
            session = None
            self.run_manifest.mark_running(prompt_id, index, prompt, workspace_dir)

            try:
                session = create_session_engine(
                    prompt=prompt,
                    workspace_dir=workspace_dir,
                    api_config=self.config["api"],
                    agent_config=self.config["agent"],
                    session_id=session_id,
                    prompt_id=prompt_id,
                    run_id=self.run_id,
                    runtime_config=self.config,
                )

                session_data = session.run()

                is_error = bool(session_data.get("error"))
                is_completed = bool(
                    session_data.get("completed") and not session_data.get("error")
                )
                status = "completed"
                if not is_completed and session_data.get("retryable"):
                    status = "retryable_error"
                elif is_error:
                    status = "fatal_error"
                elif not is_completed:
                    status = "incomplete"
                self.run_manifest.mark_result(
                    prompt_id,
                    status=status,
                    completed=is_completed,
                    retryable=bool(session_data.get("retryable")),
                    error=session_data.get("error"),
                    turns=session_data.get("turns"),
                    tool_calls_count=len(session_data.get("tool_calls") or []),
                    usage=session_data.get("usage"),
                    workspace_dir=workspace_dir,
                )

                should_retry = bool(session_data.get("retryable")) and attempt < max_attempts
                if is_error and should_retry:
                    delay = self._session_retry_delay(attempt)
                    self.logger.warning(
                        "Retryable session error for prompt %s on attempt %s/%s: %s",
                        index,
                        attempt,
                        max_attempts,
                        session_data["error"],
                    )
                    if not preserve_on_error:
                        self._cleanup_workspace(workspace_dir)
                        workspace_dir = self._create_workspace(session_id)
                    if delay > 0:
                        time.sleep(delay)
                    continue

                if is_error:
                    self.logger.error(f"Session error: {session_data['error']}")
                    if preserve_on_error:
                        self.logger.info(f"Preserving workspace: {workspace_dir}")
                    else:
                        self._cleanup_workspace(workspace_dir)

                formatted_entry = self.formatter.format_session(session_data)
                formatted_entry["tools"] = self.tool_definitions
                formatted_entry = {
                    "prompt": formatted_entry.get("prompt"),
                    "tools": formatted_entry.get("tools"),
                    "messages": formatted_entry.get("messages"),
                    "metadata": formatted_entry.get("metadata"),
                    "usage": formatted_entry.get("usage"),
                }

                if not self.formatter.validate_entry(
                    formatted_entry, require_completion=is_completed
                ):
                    self.logger.error("Entry validation failed")
                    return None

                if self.config["workspace"].get("cleanup", True) and is_completed:
                    self._cleanup_workspace(workspace_dir)
                elif not is_error:
                    self.logger.info(f"Preserving workspace: {workspace_dir}")

                return formatted_entry

            except Exception as e:
                self.logger.error(f"Error processing prompt: {e}", exc_info=True)
                self.run_manifest.mark_result(
                    prompt_id,
                    status="fatal_error",
                    completed=False,
                    retryable=False,
                    error=str(e),
                    turns=None,
                    tool_calls_count=None,
                    usage=None,
                    workspace_dir=workspace_dir,
                )
                if preserve_on_error:
                    self.logger.info(f"Preserving workspace: {workspace_dir}")
                else:
                    self._cleanup_workspace(workspace_dir)
                return None
            finally:
                if session is not None:
                    session.close()

        return None

    def _append_to_dataset(self, entry: Dict[str, Any]):
        """Append entry to dataset file."""
        jsonl_line = self.formatter.to_jsonl_line(entry)
        with self.write_lock:
            with self.output_file.open("a", encoding="utf-8") as f:
                f.write(jsonl_line + "\n")

    def _is_training_safe_entry(self, entry: Dict[str, Any]) -> bool:
        return self.formatter.is_training_safe_entry(entry)

    def _route_entry(self, entry: Dict[str, Any]) -> None:
        """Route entry to dataset if training-safe, otherwise just track in manifest."""
        prompt_id = (entry.get("metadata") or {}).get("prompt_id")
        if self._is_training_safe_entry(entry):
            self._append_to_dataset(entry)
            self.run_manifest.set_route(prompt_id, "dataset")
        elif self.formatter.is_dataset_error_entry(entry):
            self.run_manifest.set_route(prompt_id, "error_dataset")
        else:
            self.run_manifest.set_route(prompt_id, "skipped")

    def generate(self):
        """Main generation loop."""
        from tqdm import tqdm

        self.logger.info("Starting agentic dataset generation")
        self._validate_runtime_prerequisites()

        prompts = self._load_prompts()
        self.logger.info(f"Loaded {len(prompts)} prompts")

        if self.config["processing"].get("resume", True):
            completed_ids, completed_prompt_counts = self._load_completed_prompts()
            completed_total = sum(completed_prompt_counts.values())
            self.logger.info(f"Found {completed_total} clean dataset rows")
            self.run_manifest.seed_prompts(
                prompts,
                completed_ids,
                completed_prompt_counts,
            )

            remaining_prompt_counts = Counter(completed_prompt_counts)
            prompts_to_process = []
            matched_prompt_ids = 0
            matched_prompt_text_fallback = 0
            for i, prompt in enumerate(prompts):
                prompt_id = self.run_manifest.make_prompt_id(i, prompt)
                normalized_prompt = prompt.strip()
                if prompt_id in completed_ids:
                    matched_prompt_ids += 1
                    if remaining_prompt_counts.get(normalized_prompt, 0) > 0:
                        remaining_prompt_counts[normalized_prompt] -= 1
                    continue
                if remaining_prompt_counts.get(normalized_prompt, 0) > 0:
                    matched_prompt_text_fallback += 1
                    remaining_prompt_counts[normalized_prompt] -= 1
                    continue
                prompts_to_process.append((i, prompt))

            self.logger.info(
                "Resume match summary: %s prompt_ids matched directly, %s prompts matched by text fallback",
                matched_prompt_ids,
                matched_prompt_text_fallback,
            )
        else:
            self.run_manifest.seed_prompts(prompts, set())
            prompts_to_process = list(enumerate(prompts))

        if not prompts_to_process:
            self.logger.info("No prompts to process")
            self._write_dataset_readme()
            return

        self.logger.info(f"Processing {len(prompts_to_process)} prompts")

        concurrency = self.config["processing"].get("concurrency", 1)
        total_prompts = len(prompts_to_process)

        self.total_cost = 0.0
        self.total_tokens = 0

        pbar = tqdm(total=total_prompts, desc="Generating Dataset")

        def update_pbar(entry):
            if entry and "usage" in entry:
                self.total_cost += entry["usage"].get("cost", 0.0)
                self.total_tokens += entry["usage"].get("total_tokens", 0)

            pbar.set_postfix(
                {"cost": f"${self.total_cost:.4f}", "tokens": f"{self.total_tokens:,}"}
            )
            pbar.update(1)

        if concurrency <= 1:
            for index, prompt in prompts_to_process:
                entry = self._process_prompt(prompt, index)

                if entry:
                    self._route_entry(entry)
                    update_pbar(entry)
                else:
                    pbar.update(1)
        else:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = {
                    executor.submit(self._process_prompt, prompt, index): (
                        index,
                        prompt,
                    )
                    for index, prompt in prompts_to_process
                }

                for future in as_completed(futures):
                    try:
                        entry = future.result()
                        if entry:
                            self._route_entry(entry)
                            update_pbar(entry)
                    except Exception as e:
                        self.logger.error(f"Error in future: {e}")
                        pbar.update(1)

        pbar.close()
        self._write_dataset_readme()
        self.logger.info("Dataset generation complete")
        self.logger.info(f"Total Cost: ${self.total_cost:.4f}")
        self.logger.info(f"Total Tokens: {self.total_tokens:,}")
        self.logger.info(f"Output saved to: {self.output_file}")


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate agentic datasets with tool-calling capabilities"
    )
    parser.add_argument(
        "-c", "--config", required=True, help="Path to configuration YAML file"
    )

    args = parser.parse_args(argv)

    try:
        generator = AgenticDatasetGenerator(args.config)
        generator.generate()
        return 0
    except Exception as e:
        import traceback

        traceback.print_exc()
        print(f"Fatal error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

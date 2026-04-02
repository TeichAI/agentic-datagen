from __future__ import annotations

import copy
import json
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from .agent_session import AgentSession as NativeAgentSession
from .tool_registry import ToolRegistry


OPENCODE_TOOL_SPECS: dict[str, dict[str, Any]] = {
    "read": {
        "type": "function",
        "function": {
            "name": "read",
            "description": "Read file contents from the project workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filePath": {"type": "string", "description": "Path to the file to read."},
                },
                "required": ["filePath"],
            },
        },
    },
    "write": {
        "type": "function",
        "function": {
            "name": "write",
            "description": "Create a new file or overwrite an existing file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filePath": {"type": "string", "description": "Path to the file to write."},
                    "content": {"type": "string", "description": "Full file contents to write."},
                },
                "required": ["filePath", "content"],
            },
        },
    },
    "edit": {
        "type": "function",
        "function": {
            "name": "edit",
            "description": "Modify an existing file using exact string replacement.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filePath": {"type": "string", "description": "Path to the file to edit."},
                    "oldString": {"type": "string", "description": "Existing text to replace."},
                    "newString": {"type": "string", "description": "Replacement text."},
                },
                "required": ["filePath", "oldString", "newString"],
            },
        },
    },
    "list": {
        "type": "function",
        "function": {
            "name": "list",
            "description": "List files and directories in a given path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path relative to the project root."},
                },
                "required": [],
            },
        },
    },
    "glob": {
        "type": "function",
        "function": {
            "name": "glob",
            "description": "Find files by glob pattern.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Glob pattern to search for."},
                },
                "required": ["pattern"],
            },
        },
    },
    "grep": {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Search file contents using regular expressions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Pattern to search for."},
                    "path": {"type": "string", "description": "Optional path or glob scope."},
                },
                "required": ["pattern"],
            },
        },
    },
    "bash": {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command in the project environment.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to execute."},
                },
                "required": ["command"],
            },
        },
    },
    "websearch": {
        "type": "function",
        "function": {
            "name": "websearch",
            "description": "Search the web for relevant information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query."},
                },
                "required": ["query"],
            },
        },
    },
    "webfetch": {
        "type": "function",
        "function": {
            "name": "webfetch",
            "description": "Fetch the contents of a URL.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to fetch."},
                },
                "required": ["url"],
            },
        },
    },
    "todowrite": {
        "type": "function",
        "function": {
            "name": "todowrite",
            "description": "Manage a structured todo list during a coding task.",
            "parameters": {
                "type": "object",
                "properties": {
                    "todos": {
                        "type": "array",
                        "items": {"type": "object"},
                        "description": "Todo items to set or update.",
                    },
                },
                "required": ["todos"],
            },
        },
    },
}

ENABLED_TOOL_MAPPING: dict[str, list[str]] = {
    "read_file": ["read"],
    "write_file": ["write"],
    "edit_file": ["edit"],
    "list_directory": ["list", "glob"],
    "search_code": ["grep", "glob"],
    "run_command": ["bash"],
    "web_search": ["websearch", "webfetch"],
    "workspace_snapshot": ["list", "read"],
}

PROVIDER_ENV_MAPPING = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google": "GEMINI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "groq": "GROQ_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}


class BaseSessionEngine:
    def __init__(
        self,
        prompt: str,
        workspace_dir: Path,
        api_config: Dict[str, Any],
        agent_config: Dict[str, Any],
        session_id: str,
        prompt_id: Optional[str] = None,
        run_id: Optional[str] = None,
        runtime_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.prompt = prompt
        self.workspace_dir = workspace_dir
        self.api_config = api_config
        self.agent_config = agent_config
        self.session_id = session_id
        self.prompt_id = prompt_id
        self.run_id = run_id
        self.runtime_config = runtime_config or {
            "api": api_config,
            "agent": agent_config,
        }
        self.state_file = self.workspace_dir / "session_state.json"

    def _load_session_state(self) -> Optional[Dict[str, Any]]:
        if not self.state_file.exists():
            return None
        try:
            with self.state_file.open("r", encoding="utf-8") as f:
                state = json.load(f)
        except (OSError, json.JSONDecodeError):
            return None
        if state.get("prompt") != self.prompt:
            return None
        return state

    def _save_session_state(self, session_data: Dict[str, Any]) -> None:
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        with self.state_file.open("w", encoding="utf-8") as f:
            json.dump(session_data, f, ensure_ascii=False)

    def _build_session_payload(
        self,
        *,
        turn_count: int,
        messages: List[Dict[str, Any]],
        total_prompt_tokens: int,
        total_completion_tokens: int,
        total_reasoning_tokens: int,
        total_cost: float,
        final_response: Optional[str],
        completed: bool,
        tool_calls_log: List[Dict[str, Any]],
        error: Optional[str] = None,
        retryable: bool = False,
    ) -> Dict[str, Any]:
        usage = {
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "reasoning_tokens": total_reasoning_tokens,
            "total_tokens": total_prompt_tokens + total_completion_tokens + total_reasoning_tokens,
            "cost": total_cost,
        }
        return {
            "session_id": self.session_id,
            "prompt_id": self.prompt_id,
            "run_id": self.run_id,
            "prompt": self.prompt,
            "turns": turn_count,
            "conversation": messages,
            "tool_calls": tool_calls_log,
            "final_response": final_response,
            "completed": completed,
            "error": error,
            "retryable": retryable,
            "usage": usage,
        }

    def run(self) -> Dict[str, Any]:
        raise NotImplementedError

    def close(self) -> None:
        return


class NativeSessionEngine(BaseSessionEngine):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._session = NativeAgentSession(*args, **kwargs)

    def run(self) -> Dict[str, Any]:
        return self._session.run()

    def close(self) -> None:
        self._session.close()


class OpenCodeSessionEngine(BaseSessionEngine):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._tool_registry = ToolRegistry(self.workspace_dir, config=self.runtime_config)

    @staticmethod
    def _engine_config(runtime_config: Dict[str, Any]) -> Dict[str, Any]:
        agent_config = runtime_config.get("agent") or {}
        engine_config = agent_config.get("opencode") or {}
        return engine_config if isinstance(engine_config, dict) else {}

    @classmethod
    def allowed_opencode_tools(cls, enabled_tools: List[str]) -> List[str]:
        resolved: List[str] = []
        for tool_name in enabled_tools:
            resolved.extend(ENABLED_TOOL_MAPPING.get(tool_name, []))
        deduped: List[str] = []
        for tool_name in resolved:
            if tool_name not in deduped:
                deduped.append(tool_name)
        return deduped

    @classmethod
    def tool_definitions(cls, enabled_tools: List[str]) -> List[Dict[str, Any]]:
        return [
            copy.deepcopy(OPENCODE_TOOL_SPECS[tool_name])
            for tool_name in cls.allowed_opencode_tools(enabled_tools)
            if tool_name in OPENCODE_TOOL_SPECS
        ]

    def _inline_system_prompt(self) -> bool:
        engine_config = self._engine_config(self.runtime_config)
        return bool(engine_config.get("inline_system_prompt", False))

    def _runtime_prompt(self) -> str:
        system_prompt = self.agent_config.get("system_prompt")
        if not self._inline_system_prompt() or not system_prompt:
            return self.prompt
        return f"{system_prompt}\n\n{self.prompt}"

    def _command_timeout_seconds(self) -> float:
        engine_config = self._engine_config(self.runtime_config)
        timeout = engine_config.get("timeout_seconds", self.api_config.get("timeout", 600))
        try:
            return float(timeout)
        except (TypeError, ValueError):
            return 600.0

    def _binary(self) -> str:
        engine_config = self._engine_config(self.runtime_config)
        binary = engine_config.get("binary", "opencode")
        return str(binary).strip() or "opencode"

    def _model(self) -> Optional[str]:
        engine_config = self._engine_config(self.runtime_config)
        explicit = engine_config.get("model")
        if explicit:
            return str(explicit)
        model = self.api_config.get("model")
        return str(model) if model else None

    def _agent_name(self) -> Optional[str]:
        engine_config = self._engine_config(self.runtime_config)
        agent_name = engine_config.get("agent")
        return str(agent_name) if agent_name else None

    def _run_args(self) -> List[str]:
        args = [self._binary(), "run", "--format", "json"]
        model = self._model()
        if model:
            args.extend(["--model", model])
        agent_name = self._agent_name()
        if agent_name:
            args.extend(["--agent", agent_name])
        extra_args = self._engine_config(self.runtime_config).get("extra_args") or []
        if isinstance(extra_args, list):
            args.extend(str(item) for item in extra_args)
        args.append(self._runtime_prompt())
        return args

    def _permission_config(self) -> Dict[str, str]:
        allowed_tools = set(self.allowed_opencode_tools(self.agent_config.get("tools_enabled", [])))
        permissions = {tool_name: "deny" for tool_name in OPENCODE_TOOL_SPECS}
        for tool_name in allowed_tools:
            permissions[tool_name] = "allow"
        return permissions

    def _config_content(self) -> Dict[str, Any]:
        config_content: Dict[str, Any] = {
            "permission": self._permission_config(),
        }
        model = self._model()
        if model:
            config_content["model"] = model
        return config_content

    def _run_environment(self) -> Dict[str, str]:
        env = os.environ.copy()
        env["OPENCODE_CONFIG_CONTENT"] = json.dumps(self._config_content(), ensure_ascii=False)
        env["OPENCODE_DISABLE_DEFAULT_PLUGINS"] = "true"
        api_key = self.api_config.get("api_key")
        api_key_env = str(self.api_config.get("api_key_env") or "").strip()
        provider = str(self.api_config.get("provider") or "").strip().lower()
        if api_key_env and api_key:
            env[api_key_env] = str(api_key)
        provider_env = PROVIDER_ENV_MAPPING.get(provider)
        if provider_env and api_key:
            env[provider_env] = str(api_key)
        return env

    def _run_output(self) -> str:
        args = self._run_args()
        env = self._run_environment()
        timeout = self._command_timeout_seconds()
        command_runner_mode = self._tool_registry._command_runner_mode()
        if command_runner_mode == "docker":
            self._tool_registry._ensure_docker_container()
            env_prefix = " ".join(
                f"{key}={shlex.quote(str(value))}" for key, value in env.items()
            )
            command = " ".join(shlex.quote(arg) for arg in args)
            shell_command = f"env {env_prefix} {command}" if env_prefix else command
            return self._tool_registry._docker_exec_command(shell_command, timeout=timeout)
        completed = subprocess.run(
            args,
            cwd=self.workspace_dir,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if completed.returncode != 0:
            output = completed.stdout
            if completed.stderr:
                output += f"\nSTDERR:\n{completed.stderr}"
            raise RuntimeError((output or f"opencode exited with {completed.returncode}").strip())
        return completed.stdout

    @staticmethod
    def _extract_event(obj: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(obj, dict):
            return None
        payload = obj.get("payload")
        if isinstance(payload, dict) and isinstance(payload.get("type"), str):
            return payload
        if isinstance(obj.get("type"), str):
            return obj
        return None

    @classmethod
    def parse_events(cls, raw_output: str) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        for line in raw_output.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            event = cls._extract_event(parsed)
            if event is not None:
                events.append(event)
        if events:
            return events
        try:
            parsed_output = json.loads(raw_output)
        except json.JSONDecodeError as exc:
            raise RuntimeError("OpenCode did not emit parseable JSON events") from exc
        if isinstance(parsed_output, list):
            for item in parsed_output:
                event = cls._extract_event(item)
                if event is not None:
                    events.append(event)
        else:
            event = cls._extract_event(parsed_output)
            if event is not None:
                events.append(event)
        if not events:
            raise RuntimeError("OpenCode output did not contain any recognizable events")
        return events

    @staticmethod
    def _merge_part(existing: Dict[str, Any], part: Dict[str, Any], delta: Optional[str]) -> Dict[str, Any]:
        merged = dict(existing)
        for key, value in part.items():
            if key == "state" and isinstance(value, dict) and isinstance(existing.get("state"), dict):
                state = dict(existing["state"])
                state.update(value)
                merged["state"] = state
            else:
                merged[key] = value
        if delta and merged.get("type") in {"text", "reasoning"}:
            current_text = existing.get("text", "")
            merged["text"] = current_text + delta
        return merged

    @classmethod
    def build_session_data_from_events(
        cls,
        *,
        prompt: str,
        workspace_dir: Path,
        agent_config: Dict[str, Any],
        session_id: str,
        prompt_id: Optional[str],
        run_id: Optional[str],
        events: List[Dict[str, Any]],
        inline_system_prompt: bool,
    ) -> Dict[str, Any]:
        message_entries: Dict[str, Dict[str, Any]] = {}
        session_error: Optional[Dict[str, Any]] = None
        created_session_id = session_id
        for event in events:
            event_type = event.get("type")
            props = event.get("properties") or {}
            if event_type == "session.created":
                info = props.get("info") or {}
                created_session_id = info.get("id") or created_session_id
            elif event_type == "session.error":
                session_error = props.get("error") if isinstance(props, dict) else None
            elif event_type == "message.updated":
                info = props.get("info") or {}
                message_id = info.get("id")
                if not message_id:
                    continue
                entry = message_entries.setdefault(
                    message_id,
                    {"info": {}, "parts": {}, "part_order": [], "order": len(message_entries)},
                )
                entry["info"] = info
            elif event_type == "message.part.updated":
                part = props.get("part") or {}
                message_id = part.get("messageID")
                part_id = part.get("id")
                if not message_id or not part_id:
                    continue
                entry = message_entries.setdefault(
                    message_id,
                    {"info": {}, "parts": {}, "part_order": [], "order": len(message_entries)},
                )
                if part_id not in entry["parts"]:
                    entry["part_order"].append(part_id)
                entry["parts"][part_id] = cls._merge_part(
                    entry["parts"].get(part_id, {}),
                    part,
                    props.get("delta"),
                )
        system_prompt = agent_config.get("system_prompt") if inline_system_prompt else None
        conversation: List[Dict[str, Any]] = []
        if system_prompt:
            conversation.append({"role": "system", "content": str(system_prompt)})
        tool_calls_log: List[Dict[str, Any]] = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_reasoning_tokens = 0
        total_cost = 0.0
        turn_count = 0
        for entry in sorted(message_entries.values(), key=lambda item: item["order"]):
            info = entry.get("info") or {}
            role = info.get("role")
            parts = [entry["parts"][part_id] for part_id in entry.get("part_order", []) if part_id in entry["parts"]]
            if role == "user":
                user_text = "\n".join(
                    part.get("text", "")
                    for part in parts
                    if part.get("type") == "text" and isinstance(part.get("text"), str)
                ).strip()
                conversation.append({"role": "user", "content": prompt if inline_system_prompt else (user_text or prompt)})
                continue
            if role != "assistant":
                continue
            turn_count += 1
            tokens = info.get("tokens") or {}
            cache = tokens.get("cache") or {}
            total_prompt_tokens += int(tokens.get("input") or 0) + int(cache.get("read") or 0)
            total_completion_tokens += int(tokens.get("output") or 0)
            total_reasoning_tokens += int(tokens.get("reasoning") or 0)
            total_cost += float(info.get("cost") or 0.0)
            pre_reasoning: List[str] = []
            pre_text: List[str] = []
            post_reasoning: List[str] = []
            post_text: List[str] = []
            tool_calls: List[Dict[str, Any]] = []
            tool_messages: List[Dict[str, Any]] = []
            seen_tool = False
            for part in parts:
                part_type = part.get("type")
                if part_type == "reasoning":
                    text = str(part.get("text") or "")
                    if seen_tool:
                        post_reasoning.append(text)
                    else:
                        pre_reasoning.append(text)
                    continue
                if part_type == "text":
                    text = str(part.get("text") or "")
                    if seen_tool:
                        post_text.append(text)
                    else:
                        pre_text.append(text)
                    continue
                if part_type != "tool":
                    continue
                seen_tool = True
                state = part.get("state") if isinstance(part.get("state"), dict) else {}
                tool_name = str(part.get("tool") or "unknown_tool")
                call_id = str(part.get("callID") or part.get("id") or f"call_{turn_count}")
                arguments = state.get("input") if isinstance(state.get("input"), dict) else {}
                tool_calls.append(
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": arguments,
                        },
                    }
                )
                tool_output = cls._tool_output_text(state)
                tool_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": tool_name,
                        "content": tool_output,
                    }
                )
                tool_calls_log.append(
                    {
                        "turn": turn_count,
                        "tool": tool_name,
                        "arguments": arguments,
                        "result": cls._tool_result_payload(state),
                    }
                )
            if tool_calls:
                assistant_message: Dict[str, Any] = {
                    "role": "assistant",
                    "content": "\n".join(text for text in pre_text if text).strip(),
                    "tool_calls": tool_calls,
                }
                thinking = "\n".join(text for text in pre_reasoning if text).strip()
                if thinking:
                    assistant_message["thinking"] = thinking
                conversation.append(assistant_message)
                conversation.extend(tool_messages)
                final_content = "\n".join(text for text in post_text if text).strip()
                final_thinking = "\n".join(text for text in post_reasoning if text).strip()
                if final_content or final_thinking:
                    final_message: Dict[str, Any] = {
                        "role": "assistant",
                        "content": final_content,
                    }
                    if final_thinking:
                        final_message["thinking"] = final_thinking
                    conversation.append(final_message)
            else:
                assistant_message = {
                    "role": "assistant",
                    "content": "\n".join(text for text in pre_text + post_text if text).strip(),
                }
                thinking = "\n".join(text for text in pre_reasoning + post_reasoning if text).strip()
                if thinking:
                    assistant_message["thinking"] = thinking
                conversation.append(assistant_message)
            if info.get("error") and session_error is None:
                session_error = info.get("error")
        final_response = None
        completed = False
        for message in reversed(conversation):
            if message.get("role") != "assistant":
                continue
            if message.get("tool_calls"):
                continue
            final_response = message.get("content") or ""
            completed = bool(str(final_response).strip())
            break
        error_message = cls._error_message(session_error)
        retryable = cls._is_retryable_error(session_error)
        return {
            "session_id": created_session_id,
            "prompt_id": prompt_id,
            "run_id": run_id,
            "prompt": prompt,
            "turns": turn_count,
            "conversation": conversation,
            "tool_calls": tool_calls_log,
            "final_response": final_response,
            "completed": completed and error_message is None,
            "error": error_message,
            "retryable": retryable,
            "usage": {
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
                "reasoning_tokens": total_reasoning_tokens,
                "total_tokens": total_prompt_tokens + total_completion_tokens + total_reasoning_tokens,
                "cost": total_cost,
            },
        }

    @staticmethod
    def _tool_result_payload(state: Dict[str, Any]) -> Dict[str, Any]:
        payload = {"status": state.get("status")}
        if isinstance(state.get("output"), str):
            payload["output"] = state.get("output")
        if state.get("error") is not None:
            payload["error"] = state.get("error")
        if state.get("metadata") is not None:
            payload["metadata"] = state.get("metadata")
        if state.get("attachments") is not None:
            payload["attachments"] = state.get("attachments")
        return payload

    @classmethod
    def _tool_output_text(cls, state: Dict[str, Any]) -> str:
        if isinstance(state.get("output"), str) and state.get("output").strip():
            if state.get("metadata") is None and state.get("attachments") is None:
                return state.get("output")
        payload = cls._tool_result_payload(state)
        return json.dumps(payload, ensure_ascii=False)

    @staticmethod
    def _error_message(error_payload: Optional[Dict[str, Any]]) -> Optional[str]:
        if not isinstance(error_payload, dict):
            return None
        name = error_payload.get("name") or "OpenCodeError"
        data = error_payload.get("data") if isinstance(error_payload.get("data"), dict) else {}
        message = data.get("message") or data.get("responseBody") or str(data) or name
        return f"{name}: {message}"

    @staticmethod
    def _is_retryable_error(error_payload: Optional[Dict[str, Any]]) -> bool:
        if not isinstance(error_payload, dict):
            return False
        data = error_payload.get("data") if isinstance(error_payload.get("data"), dict) else {}
        if isinstance(data.get("isRetryable"), bool):
            return data["isRetryable"]
        message = str(data.get("message") or "").lower()
        retryable_tokens = ["rate", "timeout", "temporarily unavailable", "retry", "overload"]
        return any(token in message for token in retryable_tokens)

    def run(self) -> Dict[str, Any]:
        state = self._load_session_state()
        if state and state.get("completed") and not state.get("error"):
            return state
        raw_output = self._run_output()
        events = self.parse_events(raw_output)
        session_data = self.build_session_data_from_events(
            prompt=self.prompt,
            workspace_dir=self.workspace_dir,
            agent_config=self.agent_config,
            session_id=self.session_id,
            prompt_id=self.prompt_id,
            run_id=self.run_id,
            events=events,
            inline_system_prompt=self._inline_system_prompt(),
        )
        self._save_session_state(session_data)
        return session_data

    def close(self) -> None:
        self._tool_registry.close()


def create_session_engine(
    prompt: str,
    workspace_dir: Path,
    api_config: Dict[str, Any],
    agent_config: Dict[str, Any],
    session_id: str,
    prompt_id: Optional[str] = None,
    run_id: Optional[str] = None,
    runtime_config: Optional[Dict[str, Any]] = None,
) -> BaseSessionEngine:
    runtime = runtime_config or {"api": api_config, "agent": agent_config}
    engine_name = str((runtime.get("agent") or {}).get("engine", "native")).strip().lower()
    kwargs = {
        "prompt": prompt,
        "workspace_dir": workspace_dir,
        "api_config": api_config,
        "agent_config": agent_config,
        "session_id": session_id,
        "prompt_id": prompt_id,
        "run_id": run_id,
        "runtime_config": runtime,
    }
    if engine_name == "opencode":
        return OpenCodeSessionEngine(**kwargs)
    return NativeSessionEngine(**kwargs)


def get_engine_tool_definitions(
    runtime_config: Dict[str, Any], enabled_tools: List[str]
) -> List[Dict[str, Any]]:
    engine_name = str((runtime_config.get("agent") or {}).get("engine", "native")).strip().lower()
    if engine_name == "opencode":
        return OpenCodeSessionEngine.tool_definitions(enabled_tools)
    config_copy = copy.deepcopy(runtime_config)
    workspace_config = config_copy.setdefault("workspace", {})
    command_runner = workspace_config.setdefault("command_runner", {})
    command_runner["mode"] = "host"
    command_runner["eager_start"] = False
    registry = ToolRegistry(Path("."), config=config_copy)
    try:
        return registry.get_tool_definitions(enabled_tools)
    finally:
        registry.close()

import json
import re
from typing import Any, Dict, List

from .utils import prompt_requires_artifact


class Formatter:
    """Format agentic sessions to proper format."""

    THINK_BLOCK_RE = re.compile(r"^\s*<think>(.*?)</think>\s*(.*)$", re.DOTALL)

    @classmethod
    def split_content_and_thinking(
        cls,
        content: Any,
        reasoning_content: Any = None,
        thinking: Any = None,
    ) -> tuple[str, str | None]:
        normalized_content = content if isinstance(content, str) else ""
        normalized_reasoning = (
            reasoning_content.strip()
            if isinstance(reasoning_content, str) and reasoning_content.strip()
            else None
        )
        if normalized_reasoning is None and isinstance(thinking, str) and thinking.strip():
            normalized_reasoning = thinking.strip()
        if normalized_reasoning is not None:
            return normalized_content, normalized_reasoning
        if not normalized_content:
            return "", None
        match = cls.THINK_BLOCK_RE.match(normalized_content)
        if not match:
            return normalized_content, None
        extracted_thinking = match.group(1).strip()
        extracted_content = match.group(2)
        return extracted_content, extracted_thinking or None

    @staticmethod
    def _normalize_tool_call(call: Dict[str, Any]) -> Dict[str, Any]:
        normalized_call = dict(call)
        function_payload = normalized_call.get("function")
        if not isinstance(function_payload, dict):
            return normalized_call
        normalized_function = dict(function_payload)
        arguments = normalized_function.get("arguments")
        if isinstance(arguments, str):
            stripped_arguments = arguments.strip()
            if not stripped_arguments:
                normalized_function["arguments"] = {}
            else:
                try:
                    normalized_function["arguments"] = json.loads(stripped_arguments)
                except json.JSONDecodeError:
                    normalized_function["arguments"] = arguments
        elif arguments is None:
            normalized_function["arguments"] = {}
        normalized_call["function"] = normalized_function
        return normalized_call

    @classmethod
    def normalize_tool_calls(cls, tool_calls: Any) -> List[Dict[str, Any]]:
        if not isinstance(tool_calls, list):
            return []
        normalized_calls: List[Dict[str, Any]] = []
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            normalized_calls.append(cls._normalize_tool_call(tool_call))
        return normalized_calls

    @classmethod
    def canonicalize_messages(cls, messages: Any) -> List[Dict[str, Any]]:
        if not isinstance(messages, list):
            return []
        normalized_messages: List[Dict[str, Any]] = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            content, reasoning_content = cls.split_content_and_thinking(
                msg.get("content", ""),
                msg.get("reasoning_content"),
                msg.get("thinking"),
            )
            normalized_msg: Dict[str, Any] = {"role": role, "content": content}
            if reasoning_content is not None:
                normalized_msg["reasoning_content"] = reasoning_content
            if role == "assistant" and "tool_calls" in msg:
                normalized_msg["tool_calls"] = cls.normalize_tool_calls(msg.get("tool_calls"))
            if role == "tool":
                tool_call_id = msg.get("tool_call_id")
                name = msg.get("name")
                if tool_call_id is not None:
                    normalized_msg["tool_call_id"] = tool_call_id
                if name is not None:
                    normalized_msg["name"] = name
            normalized_messages.append(normalized_msg)
        return normalized_messages

    @classmethod
    def canonicalize_entry(cls, entry: Dict[str, Any]) -> Dict[str, Any]:
        normalized_entry = dict(entry)
        normalized_entry["messages"] = cls.canonicalize_messages(entry.get("messages"))
        if "tools" in entry and isinstance(entry.get("tools"), list):
            normalized_entry["tools"] = entry.get("tools")
        return normalized_entry

    @staticmethod
    def _prompt_requires_artifact(prompt: str) -> bool:
        return prompt_requires_artifact(prompt)

    @staticmethod
    def _tool_names(messages: List[Dict[str, Any]]) -> List[str]:
        tool_names: List[str] = []
        for msg in messages:
            if not isinstance(msg, dict) or msg.get("role") != "tool":
                continue
            name = msg.get("name")
            if isinstance(name, str) and name:
                tool_names.append(name)
        return tool_names

    @classmethod
    def is_suspiciously_shallow_completion(cls, entry: Dict[str, Any]) -> bool:
        if not isinstance(entry, dict):
            return False

        prompt = entry.get("prompt")
        messages = entry.get("messages")
        metadata = entry.get("metadata") or {}

        if not isinstance(prompt, str) or not prompt.strip():
            return False
        if not isinstance(messages, list) or not isinstance(metadata, dict):
            return False
        if metadata.get("completed") is not True or metadata.get("error"):
            return False
        if not cls._prompt_requires_artifact(prompt):
            return False
        if metadata.get("workspace_has_artifacts") is False:
            return True

        turns = int(metadata.get("turns") or 0)
        tool_calls_count = int(metadata.get("tool_calls_count") or 0)
        tool_names = cls._tool_names(messages)
        mutating_tools = {"write_file", "edit_file", "run_command"}
        has_mutating_tool = any(name in mutating_tools for name in tool_names)

        if has_mutating_tool:
            return False
        if tool_calls_count == 0 and turns <= 2:
            return True
        if tool_calls_count <= 1 and turns <= 3:
            return True
        return False

    @classmethod
    def is_training_safe_entry(cls, entry: Dict[str, Any]) -> bool:
        return cls.validate_entry(entry, require_completion=True) and not cls.is_suspiciously_shallow_completion(entry)

    @staticmethod
    def _metadata_error_text(entry: Dict[str, Any]) -> str:
        metadata = entry.get("metadata") or {}
        if not isinstance(metadata, dict):
            return ""
        error = metadata.get("error")
        return error.strip().lower() if isinstance(error, str) else ""

    @classmethod
    def is_max_turns_exceeded_entry(cls, entry: Dict[str, Any]) -> bool:
        return "max turns exceeded" in cls._metadata_error_text(entry)

    @classmethod
    def is_dataset_error_entry(cls, entry: Dict[str, Any]) -> bool:
        return cls.is_max_turns_exceeded_entry(entry)

    @staticmethod
    def format_session(session_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format a session into proper structure.

        The format includes:
        - conversations: multi-turn dialogue with tool calls
        - metadata about the session
        """
        conversation = session_data.get("conversation", [])
        tool_calls = session_data.get("tool_calls", [])
        usage = session_data.get("usage", {})

        formatted_messages = []

        for msg in conversation:
            role = msg.get("role")
            content, reasoning_content = Formatter.split_content_and_thinking(
                msg.get("content", ""),
                msg.get("reasoning_content"),
                msg.get("thinking"),
            )

            formatted_msg = {"role": role, "content": content}

            if reasoning_content is not None:
                formatted_msg["reasoning_content"] = reasoning_content

            if role == "assistant" and "tool_calls" in msg:
                formatted_msg["tool_calls"] = Formatter.normalize_tool_calls(
                    msg["tool_calls"]
                )

            if role == "tool":
                formatted_msg["tool_call_id"] = msg.get("tool_call_id")
                formatted_msg["name"] = msg.get("name")

            formatted_messages.append(formatted_msg)

        return {
            "prompt": session_data.get("prompt"),
            "messages": formatted_messages,
            "metadata": {
                "prompt_id": session_data.get("prompt_id"),
                "run_id": session_data.get("run_id"),
                "session_id": session_data.get("session_id"),
                "turns": session_data.get("turns"),
                "completed": session_data.get("completed", False),
                "tool_calls_count": len(tool_calls),
                "error": session_data.get("error"),
                "retryable": session_data.get("retryable", False),
                "workspace_file_count": session_data.get("workspace_file_count"),
                "workspace_has_artifacts": session_data.get("workspace_has_artifacts"),
                "successful_mutating_tool_calls": session_data.get("successful_mutating_tool_calls"),
            },
            "usage": usage,
        }

    @staticmethod
    def final_assistant_is_plain(messages: List[Dict[str, Any]]) -> bool:
        if not isinstance(messages, list) or not messages:
            return False
        last_message = messages[-1]
        if not isinstance(last_message, dict):
            return False
        if last_message.get("role") != "assistant":
            return False
        last_content = last_message.get("content")
        if not isinstance(last_content, str) or not last_content.strip():
            return False
        if "tool_calls" in last_message:
            return False
        return True

    @staticmethod
    def validate_entry(entry: Dict[str, Any], require_completion: bool = False) -> bool:
        """Validate that an entry has the required structure."""
        entry = Formatter.canonicalize_entry(entry)
        if not isinstance(entry, dict):
            return False

        prompt = entry.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            return False

        if "messages" not in entry:
            return False

        messages = entry["messages"]
        if not isinstance(messages, list) or len(messages) == 0:
            return False

        metadata = entry.get("metadata") or {}
        if not isinstance(metadata, dict):
            return False

        allowed_roles = {"system", "user", "assistant", "tool"}
        has_user = False
        has_assistant = False

        for msg in messages:
            if not isinstance(msg, dict):
                return False
            role = msg.get("role")
            if role not in allowed_roles:
                return False

            if role == "user":
                has_user = True

            if role == "assistant":
                has_assistant = True
                if "tool_calls" in msg and not isinstance(msg["tool_calls"], list):
                    return False

            if role == "tool":
                if not msg.get("tool_call_id") or not msg.get("name"):
                    return False

        if not has_user:
            return False

        if require_completion:
            if not has_assistant:
                return False
            if metadata.get("error"):
                return False
            if metadata.get("completed") is not True:
                return False
            if not Formatter.final_assistant_is_plain(messages):
                return False

        return True

    @staticmethod
    def to_jsonl_line(entry: Dict[str, Any]) -> str:
        """Convert entry to JSONL line."""
        return json.dumps(Formatter.canonicalize_entry(entry), ensure_ascii=False)

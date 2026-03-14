import json
from typing import Any, Dict, List


class Formatter:
    """Format agentic sessions to proper format."""

    @staticmethod
    def _prompt_requires_artifact(prompt: str) -> bool:
        lowered = prompt.lower()
        keywords = [
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
        ]
        return any(keyword in lowered for keyword in keywords)

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
            content = msg.get("content", "")

            formatted_msg = {"role": role, "content": content}

            if role == "assistant" and "tool_calls" in msg:
                formatted_msg["tool_calls"] = msg["tool_calls"]

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
        return json.dumps(entry, ensure_ascii=False)

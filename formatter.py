import json
from typing import Any, Dict


class Formatter:
    """Format agentic sessions to proper format."""

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
            last_message = messages[-1]
            if last_message.get("role") != "assistant":
                return False
            last_content = last_message.get("content")
            if not isinstance(last_content, str) or not last_content.strip():
                return False

        return True

    @staticmethod
    def to_jsonl_line(entry: Dict[str, Any]) -> str:
        """Convert entry to JSONL line."""
        return json.dumps(entry, ensure_ascii=False)

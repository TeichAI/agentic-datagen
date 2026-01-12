import json
from typing import Any, Dict, List


class NemotronFormatter:
    """Format agentic sessions to Nemotron-Agentic-v1 compatible format."""

    @staticmethod
    def format_session(session_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format a session into Nemotron-Agentic-v1 structure.

        The format includes:
        - conversations: multi-turn dialogue with tool calls
        - metadata about the session
        """
        conversation = session_data.get("conversation", [])
        tool_calls = session_data.get("tool_calls", [])

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
            "messages": formatted_messages,
            "metadata": {
                "session_id": session_data.get("session_id"),
                "prompt": session_data.get("prompt"),
                "turns": session_data.get("turns"),
                "completed": session_data.get("completed", False),
                "tool_calls_count": len(tool_calls),
            },
        }

    @staticmethod
    def validate_entry(entry: Dict[str, Any]) -> bool:
        """Validate that an entry has the required structure."""
        if not isinstance(entry, dict):
            return False

        if "messages" not in entry:
            return False

        messages = entry["messages"]
        if not isinstance(messages, list) or len(messages) == 0:
            return False

        for msg in messages:
            if not isinstance(msg, dict):
                return False
            if "role" not in msg:
                return False

        return True

    @staticmethod
    def to_jsonl_line(entry: Dict[str, Any]) -> str:
        """Convert entry to JSONL line."""
        return json.dumps(entry, ensure_ascii=False)

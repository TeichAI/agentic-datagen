import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from tools import ToolRegistry


class AgentSession:
    """Manages a single agentic session for one prompt."""

    def __init__(
        self,
        prompt: str,
        workspace_dir: Path,
        api_config: Dict[str, Any],
        agent_config: Dict[str, Any],
        session_id: str,
    ):
        self.prompt = prompt
        self.workspace_dir = workspace_dir
        self.api_config = api_config
        self.agent_config = agent_config
        self.session_id = session_id

        self.tool_registry = ToolRegistry(workspace_dir, config={"api": api_config})
        self.conversation_history: List[Dict[str, Any]] = []
        self.tool_calls_log: List[Dict[str, Any]] = []

        self.http_session = self._create_http_session()

    def _create_http_session(self) -> requests.Session:
        """Create HTTP session with retry logic."""
        session = requests.Session()
        retries = Retry(
            total=self.api_config.get("max_retries", 3),
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST"],
        )
        adapter = HTTPAdapter(max_retries=retries)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session

    def run(self) -> Dict[str, Any]:
        """Run the agentic session and return the complete trajectory."""
        system_prompt = self.agent_config.get("system_prompt")
        if not system_prompt:
            system_prompt = (
                "You are a helpful coding assistant with access to file operations and code analysis tools.\n"
                "Complete the user's task thoroughly and efficiently.\n"
                "When given a coding task, create working code files in the workspace."
            )

        max_turns = self.agent_config.get("max_turns") or 50
        enabled_tools = self.agent_config.get("tools_enabled", [])

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": self.prompt},
        ]

        turn_count = 0
        final_response = None
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_cost = 0.0

        while turn_count < max_turns:
            turn_count += 1

            try:
                response = self._call_llm(messages, enabled_tools)

                prompt_tokens, completion_tokens, turn_cost = self._extract_usage(
                    response
                )
                total_prompt_tokens += prompt_tokens
                total_completion_tokens += completion_tokens
                total_cost += turn_cost

            except Exception as e:
                return {
                    "session_id": self.session_id,
                    "prompt": self.prompt,
                    "error": f"LLM call failed: {str(e)}",
                    "turns": turn_count,
                    "conversation": messages,
                    "tool_calls": self.tool_calls_log,
                    "final_response": None,
                    "completed": False,
                    "usage": {
                        "prompt_tokens": total_prompt_tokens,
                        "completion_tokens": total_completion_tokens,
                        "total_tokens": total_prompt_tokens + total_completion_tokens,
                        "cost": total_cost,
                    },
                }

            assistant_message = response.get("choices", [{}])[0].get("message", {})

            if not assistant_message:
                break

            messages.append(assistant_message)

            tool_calls = assistant_message.get("tool_calls", [])

            if not tool_calls:
                final_response = assistant_message.get("content", "")
                break

            for tool_call in tool_calls:
                tool_name = tool_call.get("function", {}).get("name")
                tool_args_str = tool_call.get("function", {}).get("arguments", "{}")
                tool_id = tool_call.get("id", f"call_{turn_count}")

                try:
                    tool_args = json.loads(tool_args_str)
                except json.JSONDecodeError:
                    tool_args = {}

                tool_result = self.tool_registry.execute_tool(tool_name, tool_args)

                self.tool_calls_log.append(
                    {
                        "turn": turn_count,
                        "tool": tool_name,
                        "arguments": tool_args,
                        "result": tool_result,
                    }
                )

                result_content = json.dumps(tool_result)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_id,
                        "name": tool_name,
                        "content": result_content,
                    }
                )

        return {
            "session_id": self.session_id,
            "prompt": self.prompt,
            "turns": turn_count,
            "conversation": messages,
            "tool_calls": self.tool_calls_log,
            "final_response": final_response,
            "completed": final_response is not None,
            "usage": {
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
                "total_tokens": total_prompt_tokens + total_completion_tokens,
                "cost": total_cost,
            },
        }

    def _call_llm(
        self, messages: List[Dict[str, Any]], enabled_tools: List[str]
    ) -> Dict[str, Any]:
        """Call the LLM API."""
        api_key = self.api_config.get("api_key")
        base_url = self.api_config.get("base_url")
        model = self.api_config.get("model")
        timeout = self.api_config.get("timeout", 120)
        reasoning_effort = self.api_config.get("reasoning_effort")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        body = {
            "model": model,
            "messages": messages,
        }

        if reasoning_effort:
            body["reasoning"] = {"effort": reasoning_effort}

        if enabled_tools:
            tool_definitions = self.tool_registry.get_tool_definitions(enabled_tools)
            if tool_definitions:
                body["tools"] = tool_definitions
                body["tool_choice"] = "auto"

        response = self.http_session.post(
            base_url, headers=headers, json=body, timeout=timeout
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"API error {response.status_code}: {response.text[:500]}"
            )

        payload = response.json()
        payload["_headers"] = dict(response.headers)
        return payload

    def _extract_usage(self, response: Dict[str, Any]) -> tuple[int, int, float]:
        usage = response.get("usage") or {}
        prompt_tokens = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
        completion_tokens = (
            usage.get("completion_tokens") or usage.get("output_tokens") or 0
        )

        cost_candidates = (
            response.get("cost"),
            response.get("total_cost"),
            usage.get("cost"),
            usage.get("total_cost"),
            usage.get("total_price"),
        )
        turn_cost = next(
            (float(value) for value in cost_candidates if value is not None), 0.0
        )

        headers = response.get("_headers") or {}
        if (prompt_tokens == 0 and completion_tokens == 0) or turn_cost == 0.0:
            header_usage = headers.get("x-openrouter-usage")
            if header_usage:
                try:
                    parsed = json.loads(header_usage)
                    prompt_tokens = prompt_tokens or parsed.get("prompt_tokens", 0)
                    completion_tokens = completion_tokens or parsed.get(
                        "completion_tokens", 0
                    )
                    if turn_cost == 0.0:
                        turn_cost = parsed.get("cost", 0.0) or parsed.get(
                            "total_cost", 0.0
                        )
                except (TypeError, ValueError):
                    pass

            header_cost = headers.get("x-openrouter-cost")
            if header_cost and turn_cost == 0.0:
                try:
                    turn_cost = float(header_cost)
                except ValueError:
                    pass

        return int(prompt_tokens), int(completion_tokens), float(turn_cost)

    def close(self):
        """Clean up resources."""
        self.http_session.close()

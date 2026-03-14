import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .tools import ToolRegistry


class AgentSession:
    """Manages a single agentic session for one prompt."""

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
    ):
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

        self.tool_registry = ToolRegistry(workspace_dir, config=self.runtime_config)
        self.conversation_history: List[Dict[str, Any]] = []
        self.tool_calls_log: List[Dict[str, Any]] = []
        self.state_file = self.workspace_dir / "session_state.json"

        self.http_session = self._create_http_session()

    def _create_http_session(self) -> requests.Session:
        """Create HTTP session with retry logic."""
        session = requests.Session()
        retries = Retry(
            total=0,
            backoff_factor=0,
            status_forcelist=[],
            allowed_methods=["POST"],
        )
        adapter = HTTPAdapter(max_retries=retries)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session

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
        turn_count: int,
        messages: List[Dict[str, Any]],
        total_prompt_tokens: int,
        total_completion_tokens: int,
        total_cost: float,
        final_response: Optional[str],
        completed: bool,
        error: Optional[str] = None,
        retryable: bool = False,
    ) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "prompt_id": self.prompt_id,
            "run_id": self.run_id,
            "prompt": self.prompt,
            "turns": turn_count,
            "conversation": messages,
            "tool_calls": self.tool_calls_log,
            "final_response": final_response,
            "completed": completed,
            "error": error,
            "retryable": retryable,
            "usage": {
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
                "total_tokens": total_prompt_tokens + total_completion_tokens,
                "cost": total_cost,
            },
        }

    def _is_retryable_error(self, message: str) -> bool:
        lowered = message.lower()
        retryable_tokens = [
            "429",
            "rate limit",
            "temporarily unavailable",
            "timeout",
            "timed out",
            "connection reset",
            "connection aborted",
            "connection error",
            "service unavailable",
            "bad gateway",
            "gateway",
            "overloaded",
            "try again",
            "server error",
            "invalid json",
            "empty response",
            "malformed",
        ]
        return any(token in lowered for token in retryable_tokens)

    def _get_retry_delay(
        self, attempt: int, response: Optional[requests.Response] = None
    ) -> float:
        retry_after = None
        if response is not None:
            retry_after = response.headers.get("Retry-After")

        if retry_after:
            try:
                return max(0.0, float(retry_after))
            except ValueError:
                pass

        backoff_base = float(self.api_config.get("backoff_base_seconds", 2.0))
        backoff_max = float(self.api_config.get("backoff_max_seconds", 60.0))
        delay = min(backoff_max, backoff_base * (2 ** max(0, attempt - 1)))
        jitter = random.uniform(0, min(1.0, delay / 4 if delay else 0.25))
        return delay + jitter

    def run(self) -> Dict[str, Any]:
        """Run the agentic session and return the complete trajectory."""
        system_prompt = self.agent_config.get("system_prompt")
        if not system_prompt:
            system_prompt = (
                "You are a coding agent. Use tools deliberately, inspect before editing, "
                "and finish the user's request with working files inside the workspace. "
                "When Context7 documentation tools are available and you are working with "
                "libraries or frameworks, use Context7 to fetch the latest relevant docs "
                "before making library-specific changes."
            )

        max_turns = self.agent_config.get("max_turns") or 50
        enabled_tools = self.agent_config.get("tools_enabled", [])

        state = self._load_session_state()
        if state:
            if state.get("completed") and not state.get("error"):
                return state
            messages = state.get("conversation") or []
            if not isinstance(messages, list) or not messages:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": self.prompt},
                ]
            saved_tool_calls = state.get("tool_calls")
            if isinstance(saved_tool_calls, list):
                self.tool_calls_log = saved_tool_calls
            usage = state.get("usage") or {}
            turn_count = int(state.get("turns") or 0)
            final_response = state.get("final_response")
            total_prompt_tokens = int(usage.get("prompt_tokens") or 0)
            total_completion_tokens = int(usage.get("completion_tokens") or 0)
            total_cost = float(usage.get("cost") or 0.0)
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": self.prompt},
            ]
            turn_count = 0
            final_response = None
            total_prompt_tokens = 0
            total_completion_tokens = 0
            total_cost = 0.0
            self._save_session_state(
                self._build_session_payload(
                    turn_count=turn_count,
                    messages=messages,
                    total_prompt_tokens=total_prompt_tokens,
                    total_completion_tokens=total_completion_tokens,
                    total_cost=total_cost,
                    final_response=final_response,
                    completed=False,
                )
            )

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
                session_data = self._build_session_payload(
                    turn_count=turn_count,
                    messages=messages,
                    total_prompt_tokens=total_prompt_tokens,
                    total_completion_tokens=total_completion_tokens,
                    total_cost=total_cost,
                    final_response=None,
                    completed=False,
                    error=f"LLM call failed: {str(e)}",
                    retryable=self._is_retryable_error(str(e)),
                )
                self._save_session_state(session_data)
                return session_data

            assistant_message = response.get("choices", [{}])[0].get("message", {})

            if not assistant_message:
                session_data = self._build_session_payload(
                    turn_count=turn_count,
                    messages=messages,
                    total_prompt_tokens=total_prompt_tokens,
                    total_completion_tokens=total_completion_tokens,
                    total_cost=total_cost,
                    final_response=None,
                    completed=False,
                    error="LLM call failed: empty response message",
                    retryable=True,
                )
                self._save_session_state(session_data)
                return session_data

            # Extract reasoning/thought if present (Google Gemini / OpenRouter format)
            reasoning_content = ""
            reasoning_details = assistant_message.get("reasoning_details", [])

            # Handle list-based reasoning details (Gemini style)
            if isinstance(reasoning_details, list):
                for detail in reasoning_details:
                    if (
                        isinstance(detail, dict)
                        and detail.get("type") == "reasoning.text"
                    ):
                        reasoning_content += detail.get("text", "")
            # Handle direct string reasoning (some other providers)
            elif isinstance(reasoning_details, str):
                reasoning_content = reasoning_details

            # Also check for 'reasoning' field (DeepSeek style sometimes)
            if not reasoning_content and "reasoning" in assistant_message:
                reasoning_content = assistant_message["reasoning"]

            # Prepend reasoning to content with <think> tags
            if reasoning_content:
                original_content = assistant_message.get("content") or ""
                assistant_message["content"] = (
                    f"<think>{reasoning_content}</think>\n{original_content}"
                )

            # Sanitize message before appending to history
            # Remove provider-specific fields that might cause 400 errors on next turn
            clean_message = {
                "role": assistant_message.get("role", "assistant"),
                "content": assistant_message.get("content"),
            }
            if "tool_calls" in assistant_message:
                clean_message["tool_calls"] = assistant_message["tool_calls"]

            messages.append(clean_message)

            tool_calls = assistant_message.get("tool_calls", [])

            if not tool_calls:
                final_response = assistant_message.get("content", "")
                completed = bool(final_response and final_response.strip())
                session_data = self._build_session_payload(
                    turn_count=turn_count,
                    messages=messages,
                    total_prompt_tokens=total_prompt_tokens,
                    total_completion_tokens=total_completion_tokens,
                    total_cost=total_cost,
                    final_response=final_response,
                    completed=completed,
                    error=None if completed else "LLM call failed: empty final response",
                    retryable=not completed,
                )
                self._save_session_state(session_data)
                return session_data

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

            self._save_session_state(
                self._build_session_payload(
                    turn_count=turn_count,
                    messages=messages,
                    total_prompt_tokens=total_prompt_tokens,
                    total_completion_tokens=total_completion_tokens,
                    total_cost=total_cost,
                    final_response=final_response,
                    completed=False,
                )
            )

        session_data = self._build_session_payload(
            turn_count=turn_count,
            messages=messages,
            total_prompt_tokens=total_prompt_tokens,
            total_completion_tokens=total_completion_tokens,
            total_cost=total_cost,
            final_response=final_response,
            completed=False,
            error="LLM call failed: max turns exceeded",
            retryable=True,
        )
        self._save_session_state(session_data)
        return session_data

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

        max_attempts = max(1, int(self.api_config.get("max_retries", 3)) + 1)
        retryable_statuses = {408, 409, 425, 429, 500, 502, 503, 504}
        last_error = None

        for attempt in range(1, max_attempts + 1):
            response = None
            try:
                response = self.http_session.post(
                    base_url, headers=headers, json=body, timeout=timeout
                )
            except requests.RequestException as exc:
                last_error = f"request error: {str(exc)}"
                if attempt >= max_attempts:
                    break
                time.sleep(self._get_retry_delay(attempt))
                continue

            if response.status_code == 200:
                try:
                    payload = response.json()
                except ValueError as exc:
                    last_error = f"invalid json response: {str(exc)}"
                    if attempt >= max_attempts:
                        break
                    time.sleep(self._get_retry_delay(attempt, response))
                    continue

                choices = payload.get("choices")
                if not isinstance(choices, list) or not choices:
                    last_error = "empty response choices"
                    if attempt >= max_attempts:
                        break
                    time.sleep(self._get_retry_delay(attempt, response))
                    continue

                payload["_headers"] = dict(response.headers)
                return payload

            error_text = response.text[:500]
            last_error = f"API error {response.status_code}: {error_text}"
            if (
                response.status_code in retryable_statuses
                or self._is_retryable_error(error_text)
            ) and attempt < max_attempts:
                time.sleep(self._get_retry_delay(attempt, response))
                continue

            break

        raise RuntimeError(last_error or "unknown LLM API failure")

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
        self.tool_registry.close()
        self.http_session.close()

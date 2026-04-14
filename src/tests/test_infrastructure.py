from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from agentic_datagen.dataset_cleanup import CleanupPolicy, apply_cleanup, plan_cleanup
from agentic_datagen.dataset_qa import analyze_entry, load_reports, summarize_reports
from agentic_datagen.formatter import Formatter
from agentic_datagen.generator import AgenticDatasetGenerator
from agentic_datagen.generator import build_dataset_readme, normalize_config
from agentic_datagen.run_manifest import RunManifest
from agentic_datagen.session_engines import (
    OpenCodeSessionEngine,
    create_session_engine,
    get_engine_tool_definitions,
)
from agentic_datagen.tool_registry import ToolRegistry
from agentic_datagen.utils import (
    apply_workspace_completion_guardrails,
    get_session_state_path,
    load_guarded_session_state,
    load_prompts,
    migrate_legacy_session_state,
)


class MCPHandler(BaseHTTPRequestHandler):
    def _send(self, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8")
        payload = json.loads(raw)
        method = payload.get("method")
        request_id = payload.get("id")
        if method == "initialize":
            self._send({"jsonrpc": "2.0", "id": request_id, "result": {"capabilities": {}}})
            return
        if method == "notifications/initialized":
            self._send({})
            return
        if method == "tools/list":
            self._send(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "tools": [
                            {
                                "name": "echo",
                                "title": "Echo",
                                "description": "Echo back provided text.",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "text": {
                                            "type": "string",
                                            "description": "Text to echo.",
                                        }
                                    },
                                    "required": ["text"],
                                },
                            }
                        ]
                    },
                }
            )
            return
        if method == "tools/call":
            text = payload.get("params", {}).get("arguments", {}).get("text", "")
            self._send(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "content": [{"type": "text", "text": f"echo:{text}"}],
                        "isError": False,
                    },
                }
            )
            return
        self._send({"jsonrpc": "2.0", "id": request_id, "error": {"message": "unsupported"}})

    def log_message(self, format: str, *args) -> None:
        return


class InfrastructureTests(unittest.TestCase):
    def test_formatter_canonicalizes_inline_thinking_and_tool_arguments(self) -> None:
        formatter = Formatter()
        entry = {
            "prompt": "Build a dashboard",
            "messages": [
                {"role": "system", "content": "You are a coding agent."},
                {"role": "user", "content": "Build a dashboard"},
                {
                    "role": "assistant",
                    "content": "<think>I should create the file first.</think>",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "write_file",
                                "arguments": '{"path": "index.html", "content": "<h1>Hello</h1>"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "name": "write_file",
                    "content": '{"success": true}',
                },
                {
                    "role": "assistant",
                    "content": "<think>The file is ready.</think>Done.",
                },
            ],
            "metadata": {
                "prompt_id": "prompt_1",
                "run_id": "run_test",
                "session_id": "session_1",
                "turns": 2,
                "completed": True,
                "tool_calls_count": 1,
                "error": None,
                "retryable": False,
            },
            "usage": {"total_tokens": 10, "cost": 0.0},
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "write_file",
                        "description": "Write a file",
                        "parameters": {"type": "object"},
                    },
                }
            ],
        }

        canonical = formatter.canonicalize_entry(entry)

        self.assertEqual(
            canonical["messages"][2]["reasoning_content"],
            "I should create the file first.",
        )
        self.assertEqual(canonical["messages"][2]["content"], "")
        self.assertEqual(
            canonical["messages"][2]["tool_calls"][0]["function"]["arguments"],
            {"path": "index.html", "content": "<h1>Hello</h1>"},
        )
        self.assertEqual(
            canonical["messages"][4]["reasoning_content"],
            "The file is ready.",
        )
        self.assertEqual(canonical["messages"][4]["content"], "Done.")

    def test_normalize_config_accepts_simplified_model_and_tools_schema(self) -> None:
        config = normalize_config(
            {
                "model": {
                    "provider": "openrouter",
                    "name": "anthropic/claude-3.7-sonnet",
                    "api_key_env": "OPENROUTER_API_KEY",
                    "reasoning_effort": "high",
                    "system_prompt": "You are a coding agent.",
                    "max_turns": 25,
                },
                "prompts": {"source": "prompts.txt"},
                "tools": {
                    "enabled": ["read_file", "run_command", "web_search"],
                    "web_search": {"searxng_url": "http://localhost:8080"},
                },
                "output": {"dataset_file": "datasets/out.jsonl"},
            }
        )

        self.assertEqual(config["api"]["model"], "anthropic/claude-3.7-sonnet")
        self.assertEqual(config["api"]["reasoning_effort"], "high")
        self.assertEqual(config["api"]["searxng_url"], "http://localhost:8080")
        self.assertEqual(config["agent"]["system_prompt"], "You are a coding agent.")
        self.assertEqual(config["agent"]["max_turns"], 25)
        self.assertEqual(
            config["agent"]["tools_enabled"],
            ["read_file", "run_command", "web_search"],
        )

    def test_normalize_config_maps_openai_root_base_url_to_chat_completions(self) -> None:
        config = normalize_config(
            {
                "model": {
                    "provider": "openai",
                    "name": "gpt-4.1-mini",
                    "base_url": "http://localhost:8000/v1",
                }
            }
        )

        self.assertEqual(config["api"]["provider"], "openai")
        self.assertEqual(config["api"]["api_key_env"], "OPENAI_API_KEY")
        self.assertEqual(
            config["api"]["base_url"],
            "http://localhost:8000/v1/chat/completions",
        )

    def test_generator_allows_missing_api_key_for_local_openai_compatible_base_url(self) -> None:
        generator = AgenticDatasetGenerator.__new__(AgenticDatasetGenerator)
        generator.config = {
            "api": {
                "provider": "openai",
                "base_url": "http://127.0.0.1:8000/v1/chat/completions",
                "api_key_env": "OPENAI_API_KEY",
            }
        }
        generator.logger = unittest.mock.Mock()

        self.assertEqual(generator._get_api_key(), "")
        generator.logger.info.assert_called_once()

    def test_generator_validate_runtime_prerequisites_uses_tool_registry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            generator = AgenticDatasetGenerator.__new__(AgenticDatasetGenerator)
            generator.config = normalize_config(
                {
                    "workspace": {
                        "base_dir": tmp_dir,
                        "command_runner": {
                            "mode": "docker",
                            "docker_image": "agentic-datagen-session-runtime:latest",
                        },
                    }
                }
            )
            generator.base_workspace_dir = Path(tmp_dir)

            with patch.object(ToolRegistry, "validate_runtime_prerequisites", autospec=True) as mock_validate:
                generator._validate_runtime_prerequisites()

            mock_validate.assert_called_once()
            self.assertFalse((Path(tmp_dir) / ".runtime_probe").exists())

    def test_session_runtime_dockerfile_avoids_fragile_npm_self_upgrade(self) -> None:
        dockerfile = (Path(__file__).resolve().parents[2] / "src" / "docker" / "session-runtime.Dockerfile").read_text(
            encoding="utf-8"
        )

        self.assertNotIn("npm install -g npm@latest", dockerfile)
        self.assertIn("&& node --version \\", dockerfile)
        self.assertIn("&& npm --version \\", dockerfile)

    def test_build_dataset_readme_uses_aligned_dataset_schema(self) -> None:
        readme = build_dataset_readme(
            {
                "api": {
                    "model": "anthropic/claude-3.7-sonnet",
                    "reasoning_effort": "high",
                },
                "agent": {
                    "system_prompt": "You are Claude. A friendly, helpful assistant",
                },
                "output": {
                    "dataset_card": {
                        "title": "My Dataset",
                        "description": "Custom description",
                        "license": "apache-2.0",
                        "config_name": "default",
                        "split": "train",
                    }
                },
            },
            Path("datasets/agentic_dataset.jsonl"),
            42,
            [
                {
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "description": "Search the web",
                        "parameters": {
                            "type": "object",
                            "properties": {"query": {"type": "string"}},
                            "required": ["query"],
                        },
                    },
                }
            ],
        )

        self.assertIn("# My Dataset", readme)
        self.assertIn('pretty_name: "anthropic/claude-3.7-sonnet coding agent traces"', readme)
        self.assertIn('task_categories:', readme)
        self.assertIn('- "text-generation"', readme)
        self.assertIn('tags:', readme)
        self.assertIn('- "agent-traces"', readme)
        self.assertIn('- "coding-agent"', readme)
        self.assertIn('path: "agentic_dataset.jsonl"', readme)
        self.assertIn('"prompt": "..."', readme)
        self.assertIn('"metadata": {', readme)
        self.assertIn('"usage": {', readme)
        self.assertIn('"reasoning_content": "..."', readme)
        self.assertIn('"tool_calls"', readme)
        self.assertIn('"tools"', readme)
        self.assertIn('"tool_call_id": "call_1"', readme)
        self.assertIn('"id": "call_1"', readme)
        self.assertIn("- Reasoning effort: `high`", readme)
        self.assertIn("- Row metadata fields: `prompt_id`, `run_id`, `session_id`, `turns`, `completed`, `tool_calls_count`, `error`, `retryable`.", readme)
        self.assertIn("- Usage fields: `prompt_tokens`, `completion_tokens`, `reasoning_tokens`, `total_tokens`, `cost`.", readme)

    def test_session_engine_factory_selects_opencode_backend(self) -> None:
        engine = create_session_engine(
            prompt="Build a dashboard",
            workspace_dir=Path("."),
            api_config={"provider": "openrouter", "api_key": "test", "model": "openrouter/openai/gpt-4.1"},
            agent_config={"engine": "opencode", "tools_enabled": ["read_file", "run_command"]},
            session_id="session_1",
            runtime_config={
                "api": {"provider": "openrouter", "api_key": "test", "model": "openrouter/openai/gpt-4.1"},
                "agent": {"engine": "opencode", "tools_enabled": ["read_file", "run_command"]},
                "workspace": {"command_runner": {"mode": "host"}},
            },
        )
        try:
            self.assertIsInstance(engine, OpenCodeSessionEngine)
        finally:
            engine.close()

    def test_engine_tool_definitions_switch_to_opencode_tools(self) -> None:
        definitions = get_engine_tool_definitions(
            {
                "agent": {
                    "engine": "opencode",
                    "tools_enabled": ["read_file", "run_command", "web_search"],
                }
            },
            ["read_file", "run_command", "web_search"],
        )

        tool_names = [item["function"]["name"] for item in definitions]
        self.assertIn("read", tool_names)
        self.assertIn("bash", tool_names)
        self.assertIn("websearch", tool_names)
        self.assertIn("webfetch", tool_names)

    def test_opencode_events_are_converted_to_dataset_session_payload(self) -> None:
        session_data = OpenCodeSessionEngine.build_session_data_from_events(
            prompt="Build a dashboard",
            workspace_dir=Path("."),
            agent_config={"system_prompt": "You are a coding agent."},
            session_id="session_1",
            prompt_id="prompt_1",
            run_id="run_1",
            inline_system_prompt=False,
            events=[
                {
                    "type": "message.updated",
                    "properties": {
                        "info": {
                            "id": "user_1",
                            "sessionID": "session_1",
                            "role": "user",
                            "time": {"created": 1},
                            "agent": "coder",
                            "model": {"providerID": "openrouter", "modelID": "gpt-4.1"},
                        }
                    },
                },
                {
                    "type": "message.part.updated",
                    "properties": {
                        "part": {
                            "id": "user_part_1",
                            "sessionID": "session_1",
                            "messageID": "user_1",
                            "type": "text",
                            "text": "Build a dashboard",
                        }
                    },
                },
                {
                    "type": "message.updated",
                    "properties": {
                        "info": {
                            "id": "assistant_1",
                            "sessionID": "session_1",
                            "role": "assistant",
                            "time": {"created": 2, "completed": 3},
                            "parentID": "user_1",
                            "modelID": "gpt-4.1",
                            "providerID": "openrouter",
                            "mode": "run",
                            "path": {"cwd": ".", "root": "."},
                            "cost": 0.12,
                            "tokens": {
                                "input": 10,
                                "output": 20,
                                "reasoning": 5,
                                "cache": {"read": 0, "write": 0},
                            },
                        }
                    },
                },
                {
                    "type": "message.part.updated",
                    "properties": {
                        "part": {
                            "id": "reasoning_1",
                            "sessionID": "session_1",
                            "messageID": "assistant_1",
                            "type": "reasoning",
                            "text": "I should inspect the files first.",
                            "time": {"start": 2, "end": 2},
                        }
                    },
                },
                {
                    "type": "message.part.updated",
                    "properties": {
                        "part": {
                            "id": "tool_1",
                            "sessionID": "session_1",
                            "messageID": "assistant_1",
                            "type": "tool",
                            "callID": "call_1",
                            "tool": "read",
                            "state": {
                                "status": "completed",
                                "input": {"filePath": "src/app.py"},
                                "output": "print('hello')",
                                "title": "Read src/app.py",
                                "metadata": {},
                                "time": {"start": 2, "end": 2},
                            },
                        }
                    },
                },
                {
                    "type": "message.part.updated",
                    "properties": {
                        "part": {
                            "id": "text_1",
                            "sessionID": "session_1",
                            "messageID": "assistant_1",
                            "type": "text",
                            "text": "The dashboard is ready.",
                        }
                    },
                },
            ],
        )

        self.assertEqual(session_data["session_id"], "session_1")
        self.assertTrue(session_data["completed"])
        self.assertEqual(session_data["usage"]["reasoning_tokens"], 5)
        self.assertEqual(session_data["messages"] if "messages" in session_data else session_data["conversation"], session_data["conversation"])
        self.assertEqual(session_data["conversation"][0]["role"], "user")
        self.assertEqual(session_data["conversation"][1]["role"], "assistant")
        self.assertEqual(
            session_data["conversation"][1]["reasoning_content"],
            "I should inspect the files first.",
        )
        self.assertEqual(session_data["conversation"][1]["tool_calls"][0]["function"]["name"], "read")
        self.assertEqual(session_data["conversation"][2]["role"], "tool")
        self.assertEqual(session_data["conversation"][2]["name"], "read")
        self.assertEqual(session_data["conversation"][3]["content"], "The dashboard is ready.")
        self.assertEqual(session_data["final_response"], "The dashboard is ready.")

    def test_read_file_supports_offset_and_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir) / "workspace"
            workspace.mkdir()
            (workspace / "notes.txt").write_text("alpha\nbeta\ngamma\ndelta\n", encoding="utf-8")
            registry = ToolRegistry(
                workspace,
                config={"workspace": {"command_runner": {"mode": "host"}}},
            )

            self.assertEqual(registry.read_file("notes.txt", offset=2, limit=2), "beta\ngamma\n")
            self.assertEqual(registry.read_file("notes.txt", offset=3), "gamma\ndelta\n")
            self.assertEqual(registry.read_file("notes.txt", limit=1), "alpha\n")

    @patch.object(ToolRegistry, "_ensure_docker_container", autospec=True)
    def test_tool_registry_defaults_to_lazy_docker_isolation(self, mock_ensure_docker_container) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir) / "workspace"
            workspace.mkdir()

            registry = ToolRegistry(workspace)
            try:
                self.assertEqual(registry._command_runner_mode(), "docker")
                self.assertEqual(registry._command_runner_tool_scope(), "all")
                mock_ensure_docker_container.assert_not_called()
            finally:
                registry.close()

    def test_legacy_session_state_is_migrated_out_of_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir) / "workspace"
            workspace.mkdir()
            legacy_state = workspace / "session_state.json"
            legacy_state.write_text('{"prompt": "Build a dashboard"}', encoding="utf-8")

            external_state = get_session_state_path(workspace)
            if external_state.exists():
                external_state.unlink()

            migrate_legacy_session_state(workspace, external_state)

            registry = ToolRegistry(
                workspace,
                config={"workspace": {"command_runner": {"mode": "host"}}},
            )
            self.assertFalse(legacy_state.exists())
            self.assertTrue(external_state.exists())
            self.assertEqual(registry.list_directory(), [])
            with self.assertRaises(FileNotFoundError):
                registry.read_file("session_state.json")

    def test_cleanup_workspace_removes_external_session_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir) / "workspace"
            workspace.mkdir()
            (workspace / "hello.txt").write_text("hello", encoding="utf-8")
            external_state = get_session_state_path(workspace)
            external_state.parent.mkdir(parents=True, exist_ok=True)
            external_state.write_text('{"prompt": "Build a dashboard"}', encoding="utf-8")

            generator = AgenticDatasetGenerator.__new__(AgenticDatasetGenerator)
            generator.logger = unittest.mock.Mock()

            generator._cleanup_workspace(workspace)

            self.assertFalse(workspace.exists())
            self.assertFalse(external_state.exists())

    def test_load_guarded_session_state_discards_stale_empty_workspace_terminal_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir) / "workspace"
            workspace.mkdir()
            external_state = get_session_state_path(workspace)
            external_state.parent.mkdir(parents=True, exist_ok=True)
            external_state.write_text(
                json.dumps(
                    {
                        "session_id": "session_000003",
                        "prompt_id": "prompt_000003",
                        "run_id": "run_old",
                        "prompt": "Build a dashboard",
                        "turns": 3,
                        "conversation": [{"role": "assistant", "content": "Done."}],
                        "tool_calls": [
                            {
                                "turn": 1,
                                "tool": "write_file",
                                "arguments": {"file_path": "index.html"},
                                "result": {"success": True, "result": "ok"},
                            }
                        ],
                        "final_response": "Done.",
                        "completed": True,
                        "error": None,
                        "retryable": False,
                        "usage": {"total_tokens": 10, "cost": 0.0},
                    }
                ),
                encoding="utf-8",
            )

            loaded = load_guarded_session_state(
                external_state,
                prompt="Build a dashboard",
                workspace_dir=workspace,
            )

            self.assertIsNone(loaded)
            self.assertFalse(external_state.exists())

    def test_workspace_completion_guardrails_demote_completed_artifact_session_without_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir) / "workspace"
            workspace.mkdir()

            session_data = {
                "session_id": "session_guardrail",
                "prompt_id": "prompt_guardrail",
                "run_id": "run_test",
                "prompt": "Build a dashboard",
                "turns": 3,
                "conversation": [{"role": "assistant", "content": "Done."}],
                "tool_calls": [
                    {
                        "turn": 1,
                        "tool": "write_file",
                        "arguments": {"file_path": "index.html"},
                        "result": {"success": True, "result": "ok"},
                    }
                ],
                "final_response": "Done.",
                "completed": True,
                "error": None,
                "retryable": False,
                "usage": {"total_tokens": 10, "cost": 0.0},
            }

            guarded = apply_workspace_completion_guardrails(
                session_data,
                prompt="Build a dashboard",
                workspace_dir=workspace,
            )

            self.assertFalse(guarded["completed"])
            self.assertEqual(guarded["workspace_file_count"], 0)
            self.assertFalse(guarded["workspace_has_artifacts"])
            self.assertEqual(guarded["successful_mutating_tool_calls"], 1)
            self.assertIn("workspace has no preserved files", guarded["error"])

    def test_workspace_completion_guardrails_keep_completed_artifact_session_with_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir) / "workspace"
            workspace.mkdir()
            (workspace / "index.html").write_text("<h1>ready</h1>\n", encoding="utf-8")

            session_data = {
                "session_id": "session_guardrail_ok",
                "prompt_id": "prompt_guardrail_ok",
                "run_id": "run_test",
                "prompt": "Build a dashboard",
                "turns": 2,
                "conversation": [{"role": "assistant", "content": "Done."}],
                "tool_calls": [
                    {
                        "turn": 1,
                        "tool": "write_file",
                        "arguments": {"file_path": "index.html"},
                        "result": {"success": True, "result": "ok"},
                    }
                ],
                "final_response": "Done.",
                "completed": True,
                "error": None,
                "retryable": False,
                "usage": {"total_tokens": 8, "cost": 0.0},
            }

            guarded = apply_workspace_completion_guardrails(
                session_data,
                prompt="Build a dashboard",
                workspace_dir=workspace,
            )

            self.assertTrue(guarded["completed"])
            self.assertIsNone(guarded["error"])
            self.assertEqual(guarded["workspace_file_count"], 1)
            self.assertTrue(guarded["workspace_has_artifacts"])
            self.assertEqual(guarded["successful_mutating_tool_calls"], 1)

    def test_formatter_rejects_suspiciously_shallow_build_completion(self) -> None:
        formatter = Formatter()
        shallow_entry = {
            "prompt": "Build a professional landing page for a private security firm that feels safe and protective.",
            "messages": [
                {"role": "system", "content": "You are a coding agent."},
                {"role": "user", "content": "Build a professional landing page for a private security firm that feels safe and protective."},
                {
                    "role": "assistant",
                    "content": "<think>Let me inspect the workspace first.</think>",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "list_directory",
                                "arguments": "{}",
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "name": "list_directory",
                    "content": '{"success": true, "result": ["session_state.json"], "tool": "list_directory", "source": "builtin"}',
                },
                {
                    "role": "assistant",
                    "content": "<think>The workspace is empty.</think> I'll create a professional landing page for the firm.",
                },
            ],
            "metadata": {
                "prompt_id": "prompt_shallow",
                "run_id": "run_test",
                "session_id": "session_shallow",
                "turns": 2,
                "completed": True,
                "tool_calls_count": 1,
                "error": None,
                "retryable": False,
            },
            "usage": {"total_tokens": 10, "cost": 0.0},
        }
        substantive_entry = {
            "prompt": "Build a professional landing page for a private security firm that feels safe and protective.",
            "messages": [
                {"role": "system", "content": "You are a coding agent."},
                {"role": "user", "content": "Build a professional landing page for a private security firm that feels safe and protective."},
                {
                    "role": "assistant",
                    "content": "<think>I'll create the landing page.</think>",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "write_file",
                                "arguments": "{}",
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "name": "write_file",
                    "content": '{"success": true, "result": "ok", "tool": "write_file", "source": "builtin"}',
                },
                {
                    "role": "assistant",
                    "content": "<think>The file is written.</think> The landing page is ready.",
                },
            ],
            "metadata": {
                "prompt_id": "prompt_substantive",
                "run_id": "run_test",
                "session_id": "session_substantive",
                "turns": 2,
                "completed": True,
                "tool_calls_count": 1,
                "error": None,
                "retryable": False,
            },
            "usage": {"total_tokens": 10, "cost": 0.0},
        }

        self.assertTrue(formatter.is_suspiciously_shallow_completion(shallow_entry))
        self.assertFalse(formatter.is_training_safe_entry(shallow_entry))
        self.assertFalse(formatter.is_suspiciously_shallow_completion(substantive_entry))
        self.assertTrue(formatter.is_training_safe_entry(substantive_entry))

    def test_formatter_rejects_completed_artifact_prompt_without_workspace_artifacts(self) -> None:
        formatter = Formatter()
        entry = {
            "prompt": "Build a polished landing page for a SaaS dashboard.",
            "messages": [
                {"role": "system", "content": "You are a coding agent."},
                {"role": "user", "content": "Build a polished landing page for a SaaS dashboard."},
                {
                    "role": "assistant",
                    "content": "<think>I will write the page now.</think>",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "write_file",
                                "arguments": {},
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "name": "write_file",
                    "content": '{"success": true, "result": "ok"}',
                },
                {
                    "role": "assistant",
                    "content": "<think>The page is done.</think>Finished.",
                },
            ],
            "metadata": {
                "prompt_id": "prompt_missing_artifacts",
                "run_id": "run_test",
                "session_id": "session_missing_artifacts",
                "turns": 2,
                "completed": True,
                "tool_calls_count": 1,
                "error": None,
                "retryable": False,
                "workspace_file_count": 0,
                "workspace_has_artifacts": False,
                "successful_mutating_tool_calls": 1,
            },
            "usage": {"total_tokens": 10, "cost": 0.0},
        }

        self.assertTrue(formatter.is_suspiciously_shallow_completion(entry))
        self.assertFalse(formatter.is_training_safe_entry(entry))

    def test_formatter_rejects_final_assistant_with_tool_calls(self) -> None:
        formatter = Formatter()
        invalid_entry = {
            "prompt": "Build a dashboard",
            "messages": [
                {"role": "system", "content": "You are a coding agent."},
                {"role": "user", "content": "Build a dashboard"},
                {
                    "role": "assistant",
                    "content": "Calling a tool.",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "write_file",
                                "arguments": "{}",
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "name": "write_file",
                    "content": '{"success": true, "result": "ok", "tool": "write_file", "source": "builtin"}',
                },
                {
                    "role": "assistant",
                    "content": "I am done but still requesting another tool.",
                    "tool_calls": [
                        {
                            "id": "call_2",
                            "type": "function",
                            "function": {
                                "name": "run_command",
                                "arguments": "{}",
                            },
                        }
                    ],
                },
            ],
            "metadata": {
                "prompt_id": "prompt_bad_final_assistant",
                "run_id": "run_test",
                "session_id": "session_bad_final_assistant",
                "turns": 3,
                "completed": True,
                "tool_calls_count": 2,
                "error": None,
                "retryable": False,
            },
            "usage": {"total_tokens": 10, "cost": 0.0},
        }

        self.assertFalse(formatter.final_assistant_is_plain(invalid_entry["messages"]))
        self.assertFalse(formatter.validate_entry(invalid_entry, require_completion=True))
        self.assertFalse(formatter.is_training_safe_entry(invalid_entry))

        report = analyze_entry(invalid_entry, 1, formatter=formatter)
        self.assertIn("final_assistant_has_tool_calls", report["errors"])
        self.assertFalse(report["quality"]["ended_with_plain_assistant"])

    def test_dataset_qa_summarizes_structural_and_tool_failures(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_path = Path(tmp_dir) / "dataset.jsonl"
            valid_entry = {
                "prompt": "Build a dashboard",
                "messages": [
                    {"role": "system", "content": "You are a coding agent."},
                    {"role": "user", "content": "Build a dashboard"},
                    {
                        "role": "assistant",
                        "content": "<think>I'll write the dashboard file.</think>",
                        "tool_calls": [
                            {
                                "id": "call_valid_1",
                                "type": "function",
                                "function": {
                                    "name": "write_file",
                                    "arguments": "{}",
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_valid_1",
                        "name": "write_file",
                        "content": '{"success": true, "result": "ok", "tool": "write_file", "source": "builtin"}',
                    },
                    {"role": "assistant", "content": "<think>The file is written.</think>Done."},
                ],
                "metadata": {
                    "prompt_id": "prompt_valid",
                    "run_id": "run_test",
                    "session_id": "session_valid",
                    "turns": 2,
                    "completed": True,
                    "tool_calls_count": 1,
                    "error": None,
                    "retryable": False,
                },
                "usage": {"total_tokens": 10, "cost": 0.0},
            }
            flagged_entry = {
                "prompt": "Build an API",
                "messages": [
                    {"role": "system", "content": "You are a coding agent."},
                    {"role": "user", "content": "Build an API"},
                    {
                        "role": "assistant",
                        "content": "Calling a tool now.",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "write_file",
                                    "arguments": "{}",
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_1",
                        "name": "write_file",
                        "content": '{"success": false, "error": "disk full"}',
                    },
                ],
                "metadata": {
                    "prompt_id": "prompt_flagged",
                    "run_id": "run_test",
                    "session_id": "session_flagged",
                    "turns": 1,
                    "completed": False,
                    "tool_calls_count": 1,
                    "error": "tool failed",
                    "retryable": False,
                },
                "usage": {"total_tokens": 8, "cost": 0.0},
            }
            dataset_path.write_text(
                "\n".join(
                    [
                        json.dumps(valid_entry, ensure_ascii=False),
                        json.dumps(flagged_entry, ensure_ascii=False),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            reports = load_reports(dataset_path)
            summary = summarize_reports(reports, dataset_path=dataset_path)

            self.assertEqual(len(reports), 2)
            self.assertEqual(summary["totals"]["entries"], 2)
            self.assertEqual(summary["totals"]["training_safe"], 1)
            self.assertEqual(summary["totals"]["ended_with_assistant"], 1)
            self.assertEqual(summary["totals"]["metadata_error_entries"], 1)
            self.assertEqual(summary["totals"]["entries_with_failed_tool_calls"], 1)
            self.assertEqual(summary["totals"]["total_failed_tool_calls"], 1)
            self.assertEqual(summary["issue_counts"]["errors"]["metadata_error"], 1)
            self.assertEqual(
                summary["issue_counts"]["errors"]["last_message_not_assistant"],
                1,
            )
            self.assertEqual(
                summary["issue_counts"]["warnings"]["failed_tool_calls_present"],
                1,
            )
            self.assertEqual(summary["issue_counts"]["info"], {})
            self.assertEqual(summary["totals"]["final_rows"], 2)
            self.assertEqual(summary["totals"]["total_tokens"], 18)
            self.assertEqual(summary["totals"]["average_turns"], 1.5)

    def test_load_prompts_preserves_duplicate_jsonl_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            prompts_path = Path(tmp_dir) / "prompts.jsonl"
            payloads = [
                {"messages": [{"role": "user", "content": "alpha"}]},
                {"messages": [{"role": "user", "content": "beta"}]},
                {"messages": [{"role": "user", "content": "alpha"}]},
            ]
            prompts_path.write_text(
                "\n".join(json.dumps(item, ensure_ascii=False) for item in payloads) + "\n",
                encoding="utf-8",
            )

            prompts = load_prompts(prompts_path)

            self.assertEqual(prompts, ["alpha", "beta", "alpha"])

    def test_run_manifest_uses_prompt_id_for_duplicate_prompt_completion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "run_manifest.json"
            manifest = RunManifest(path, run_id="test-run")
            prompts = ["alpha", "beta", "alpha"]
            completed_prompt_id = manifest.make_prompt_id(0, "alpha")

            manifest.seed_prompts(prompts, {completed_prompt_id})
            snapshot = manifest.snapshot()

            first_alpha_id = manifest.make_prompt_id(0, "alpha")
            second_alpha_id = manifest.make_prompt_id(2, "alpha")
            self.assertEqual(snapshot["entries"][first_alpha_id]["status"], "completed")
            self.assertEqual(snapshot["entries"][second_alpha_id]["status"], "pending")

    def test_tool_registry_accepts_compatibility_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir) / "workspace"
            workspace.mkdir()
            (workspace / "nested").mkdir()
            (workspace / "nested" / "hello.txt").write_text("hello", encoding="utf-8")
            registry = ToolRegistry(
                workspace,
                config={"workspace": {"command_runner": {"mode": "host"}}},
            )

            listing = registry.list_directory(file_path="nested")
            output = registry.run_command(
                f'"{sys.executable}" -c "print(\'hello\', end=\'\')"',
                timeout=1,
            )

            self.assertEqual(listing, ["nested/hello.txt (5 bytes)"])
            self.assertEqual(output, "hello")

    def test_dataset_qa_classifies_localhost_as_info(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_path = Path(tmp_dir) / "dataset.jsonl"
            entry = {
                "prompt": "Build a dashboard",
                "messages": [
                    {"role": "system", "content": "You are a coding agent."},
                    {"role": "user", "content": "Build a dashboard"},
                    {
                        "role": "assistant",
                        "content": "<think>I will write the dashboard file.</think>",
                        "tool_calls": [
                            {
                                "id": "call_info_1",
                                "type": "function",
                                "function": {
                                    "name": "write_file",
                                    "arguments": "{}",
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_info_1",
                        "name": "write_file",
                        "content": '{"success": true, "result": "ok", "tool": "write_file", "source": "builtin"}',
                    },
                    {
                        "role": "assistant",
                        "content": "<think>I launched the preview.</think> The app is available at http://localhost:3000.",
                    },
                ],
                "metadata": {
                    "prompt_id": "prompt_info",
                    "run_id": "run_test",
                    "session_id": "session_info",
                    "turns": 2,
                    "completed": True,
                    "tool_calls_count": 1,
                    "error": None,
                    "retryable": False,
                },
                "usage": {"total_tokens": 4, "cost": 0.0},
            }
            dataset_path.write_text(json.dumps(entry, ensure_ascii=False) + "\n", encoding="utf-8")

            reports = load_reports(dataset_path)
            summary = summarize_reports(reports, dataset_path=dataset_path)

            self.assertEqual(summary["issue_counts"]["warnings"], {})
            self.assertEqual(summary["issue_counts"]["info"]["mentions_localhost"], 1)

    def test_dataset_qa_flags_max_turns_exceeded_explicitly(self) -> None:
        formatter = Formatter()
        entry = {
            "prompt": "Build a dashboard",
            "messages": [
                {"role": "system", "content": "You are a coding agent."},
                {"role": "user", "content": "Build a dashboard"},
                {"role": "assistant", "content": "I am still working on it."},
            ],
            "metadata": {
                "prompt_id": "prompt_max_turns",
                "run_id": "run_test",
                "session_id": "session_max_turns",
                "turns": 50,
                "completed": False,
                "tool_calls_count": 0,
                "error": "LLM call failed: max turns exceeded",
                "retryable": True,
            },
            "usage": {"total_tokens": 12, "cost": 0.0},
        }

        report = analyze_entry(entry, 1, formatter=formatter)

        self.assertIn("metadata_error", report["errors"])
        self.assertIn("max_turns_exceeded", report["errors"])
        self.assertFalse(report["quality"]["training_safe"])

    def test_dataset_cleanup_plans_and_rewrites(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_path = Path(tmp_dir) / "dataset.jsonl"
            backup_path = Path(tmp_dir) / "dataset.backup.jsonl"
            good_entry = {
                "prompt": "Build a dashboard",
                "messages": [
                    {"role": "system", "content": "You are a coding agent."},
                    {"role": "user", "content": "Build a dashboard"},
                    {
                        "role": "assistant",
                        "content": "<think>I will write the file.</think>",
                        "tool_calls": [
                            {
                                "id": "call_good",
                                "type": "function",
                                "function": {"name": "write_file", "arguments": "{}"},
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_good",
                        "name": "write_file",
                        "content": '{"success": true, "result": "ok"}',
                    },
                    {"role": "assistant", "content": "<think>Done.</think>Finished."},
                ],
                "metadata": {
                    "prompt_id": "prompt_good",
                    "run_id": "run_test",
                    "session_id": "session_good",
                    "turns": 2,
                    "completed": True,
                    "tool_calls_count": 1,
                    "error": None,
                    "retryable": False,
                },
                "usage": {"total_tokens": 5, "cost": 0.0},
            }
            shallow_entry = {
                "prompt": "Build a landing page",
                "messages": [
                    {"role": "system", "content": "You are a coding agent."},
                    {"role": "user", "content": "Build a landing page"},
                    {"role": "assistant", "content": "<think>I will build it.</think>"},
                ],
                "metadata": {
                    "prompt_id": "prompt_shallow",
                    "run_id": "run_test",
                    "session_id": "session_shallow",
                    "turns": 1,
                    "completed": True,
                    "tool_calls_count": 0,
                    "error": None,
                    "retryable": False,
                },
                "usage": {"total_tokens": 3, "cost": 0.0},
            }
            dataset_path.write_text(
                "\n".join(
                    [
                        json.dumps(good_entry, ensure_ascii=False),
                        json.dumps(shallow_entry, ensure_ascii=False),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            rows = [
                {"index": 1, "raw": json.dumps(good_entry, ensure_ascii=False)},
                {"index": 2, "raw": json.dumps(shallow_entry, ensure_ascii=False)},
            ]
            reports = load_reports(dataset_path)
            policy = CleanupPolicy(remove_errors=True, remove_shallow=True, remove_port_conflicts=True)
            plan = plan_cleanup(rows, reports, policy)

            self.assertEqual(plan["removed_count"], 1)
            self.assertEqual(plan["retained_count"], 1)
            self.assertEqual(plan["removal_reason_counts"]["suspiciously_shallow_completion"], 1)

            written_backup = apply_cleanup(dataset_path, plan, backup_path=backup_path)
            self.assertEqual(written_backup, backup_path)
            rewritten_lines = [line for line in dataset_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(len(rewritten_lines), 1)
            self.assertIn("prompt_good", rewritten_lines[0])
            self.assertTrue(backup_path.exists())

    @patch("agentic_datagen.tool_registry.subprocess.run")
    def test_run_command_uses_docker_runner_when_configured(self, mock_run) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir) / "workspace"
            workspace.mkdir()
            mock_run.side_effect = [
                subprocess.CompletedProcess(args=["docker", "run"], returncode=0, stdout="container-id\n", stderr=""),
                subprocess.CompletedProcess(args=["docker", "exec"], returncode=0, stdout="hello from container\n", stderr=""),
                subprocess.CompletedProcess(args=["docker", "rm"], returncode=0, stdout="", stderr=""),
            ]
            registry = ToolRegistry(
                workspace,
                config={
                    "workspace": {
                        "command_runner": {
                            "mode": "docker",
                            "docker_image": "python:3.12-slim",
                            "container_workspace_dir": "/workspace",
                        }
                    }
                },
            )

            output = registry.run_command("python -V")
            registry.close()

            self.assertIn("hello from container", output)
            docker_run_args = mock_run.call_args_list[0].args[0]
            docker_exec_args = mock_run.call_args_list[1].args[0]
            docker_rm_args = mock_run.call_args_list[2].args[0]
            self.assertEqual(docker_run_args[:3], ["docker", "run", "-d"])
            self.assertEqual(docker_exec_args[:2], ["docker", "exec"])
            self.assertEqual(docker_rm_args[:3], ["docker", "rm", "-f"])

    @patch("agentic_datagen.tool_registry.subprocess.run")
    def test_run_command_reports_missing_default_docker_image_clearly(self, mock_run) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir) / "workspace"
            workspace.mkdir()
            mock_run.side_effect = [
                subprocess.CompletedProcess(
                    args=["docker", "run"],
                    returncode=125,
                    stdout="",
                    stderr=(
                        "Unable to find image 'agentic-datagen-session-runtime:latest' locally\n"
                        "docker: Error response from daemon: pull access denied for "
                        "agentic-datagen-session-runtime, repository does not exist or may require 'docker login'."
                    ),
                )
            ]
            registry = ToolRegistry(workspace)

            with self.assertRaises(RuntimeError) as exc_info:
                registry.run_command("python -V")

            message = str(exc_info.exception)
            self.assertIn("Docker image 'agentic-datagen-session-runtime:latest' is not available.", message)
            self.assertIn(
                "docker build -t agentic-datagen-session-runtime:latest -f src/docker/session-runtime.Dockerfile .",
                message,
            )

    @patch("agentic_datagen.tool_registry.subprocess.run")
    def test_validate_runtime_prerequisites_reports_missing_default_docker_image_clearly(self, mock_run) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir) / "workspace"
            workspace.mkdir()
            mock_run.return_value = subprocess.CompletedProcess(
                args=["docker", "image", "inspect"],
                returncode=1,
                stdout="",
                stderr=(
                    "Error response from daemon: No such image: "
                    "agentic-datagen-session-runtime:latest"
                ),
            )
            registry = ToolRegistry(workspace)

            with self.assertRaises(RuntimeError) as exc_info:
                registry.validate_runtime_prerequisites()

            self.assertIn("Docker image 'agentic-datagen-session-runtime:latest' is not available.", str(exc_info.exception))

    @patch("agentic_datagen.tool_registry.subprocess.run")
    def test_workspace_tools_use_docker_when_full_scope_configured(self, mock_run) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir) / "workspace"
            workspace.mkdir()
            mock_run.side_effect = [
                subprocess.CompletedProcess(args=["docker", "run"], returncode=0, stdout="container-id\n", stderr=""),
                subprocess.CompletedProcess(args=["docker", "exec"], returncode=0, stdout="Successfully wrote 5 characters to hello.txt", stderr=""),
                subprocess.CompletedProcess(args=["docker", "inspect"], returncode=0, stdout="true\n", stderr=""),
                subprocess.CompletedProcess(args=["docker", "exec"], returncode=0, stdout="hello", stderr=""),
                subprocess.CompletedProcess(args=["docker", "inspect"], returncode=0, stdout="true\n", stderr=""),
                subprocess.CompletedProcess(args=["docker", "exec"], returncode=0, stdout='["hello.txt (5 bytes)"]', stderr=""),
                subprocess.CompletedProcess(args=["docker", "rm"], returncode=0, stdout="", stderr=""),
            ]
            registry = ToolRegistry(
                workspace,
                config={
                    "workspace": {
                        "command_runner": {
                            "mode": "docker",
                            "tool_scope": "all",
                            "docker_image": "node:22-bookworm",
                            "container_workspace_dir": "/workspace",
                        }
                    }
                },
            )

            write_result = registry.write_file("hello.txt", "hello")
            read_result = registry.read_file("hello.txt")
            listing = registry.list_directory()
            registry.close()

            self.assertIn("Successfully wrote 5 characters", write_result)
            self.assertEqual(read_result, "hello")
            self.assertEqual(listing, ["hello.txt (5 bytes)"])
            docker_run_args = mock_run.call_args_list[0].args[0]
            docker_write_args = mock_run.call_args_list[1].args[0]
            docker_read_args = mock_run.call_args_list[3].args[0]
            docker_list_args = mock_run.call_args_list[5].args[0]
            docker_rm_args = mock_run.call_args_list[6].args[0]
            self.assertEqual(docker_run_args[:3], ["docker", "run", "-d"])
            self.assertEqual(docker_write_args[:3], ["docker", "exec", "-i"])
            self.assertEqual(docker_read_args[:3], ["docker", "exec", "-i"])
            self.assertEqual(docker_list_args[:3], ["docker", "exec", "-i"])
            self.assertEqual(docker_rm_args[:3], ["docker", "rm", "-f"])

    @patch("agentic_datagen.tool_registry.subprocess.run")
    def test_docker_bootstrap_runs_once_before_first_command(self, mock_run) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir) / "workspace"
            workspace.mkdir()
            mock_run.side_effect = [
                subprocess.CompletedProcess(args=["docker", "run"], returncode=0, stdout="container-id\n", stderr=""),
                subprocess.CompletedProcess(args=["docker", "exec"], returncode=0, stdout="bootstrapped\n", stderr=""),
                subprocess.CompletedProcess(args=["docker", "exec"], returncode=0, stdout="hello\n", stderr=""),
                subprocess.CompletedProcess(args=["docker", "inspect"], returncode=0, stdout="true\n", stderr=""),
                subprocess.CompletedProcess(args=["docker", "exec"], returncode=0, stdout="again\n", stderr=""),
                subprocess.CompletedProcess(args=["docker", "rm"], returncode=0, stdout="", stderr=""),
            ]
            registry = ToolRegistry(
                workspace,
                config={
                    "workspace": {
                        "command_runner": {
                            "mode": "docker",
                            "docker_image": "node:22-bookworm",
                            "container_workspace_dir": "/workspace",
                            "bootstrap_commands": ["echo bootstrapped"],
                            "bootstrap_trigger": "before_first_command",
                        }
                    }
                },
            )

            first_output = registry.run_command("echo hello")
            second_output = registry.run_command("echo again")
            registry.close()

            self.assertIn("hello", first_output)
            self.assertIn("again", second_output)
            bootstrap_args = mock_run.call_args_list[1].args[0]
            first_command_args = mock_run.call_args_list[2].args[0]
            second_command_args = mock_run.call_args_list[4].args[0]
            self.assertEqual(bootstrap_args[:3], ["docker", "exec", "-i"])
            self.assertEqual(first_command_args[:3], ["docker", "exec", "-i"])
            self.assertEqual(second_command_args[:3], ["docker", "exec", "-i"])
            self.assertIn("echo bootstrapped", bootstrap_args[-1])
            self.assertIn("echo hello", first_command_args[-1])
            self.assertIn("echo again", second_command_args[-1])

    @patch("agentic_datagen.tool_registry.subprocess.run")
    def test_docker_bootstrap_can_run_on_container_start(self, mock_run) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir) / "workspace"
            workspace.mkdir()
            mock_run.side_effect = [
                subprocess.CompletedProcess(args=["docker", "run"], returncode=0, stdout="container-id\n", stderr=""),
                subprocess.CompletedProcess(args=["docker", "exec"], returncode=0, stdout="bootstrapped\n", stderr=""),
                subprocess.CompletedProcess(args=["docker", "rm"], returncode=0, stdout="", stderr=""),
            ]

            registry = ToolRegistry(
                workspace,
                config={
                    "workspace": {
                        "command_runner": {
                            "mode": "docker",
                            "docker_image": "node:22-bookworm",
                            "container_workspace_dir": "/workspace",
                            "eager_start": True,
                            "bootstrap_commands": ["echo bootstrapped"],
                            "bootstrap_trigger": "container_start",
                        }
                    }
                },
            )
            registry.close()

            docker_run_args = mock_run.call_args_list[0].args[0]
            bootstrap_args = mock_run.call_args_list[1].args[0]
            docker_rm_args = mock_run.call_args_list[2].args[0]
            self.assertEqual(docker_run_args[:3], ["docker", "run", "-d"])
            self.assertEqual(bootstrap_args[:3], ["docker", "exec", "-i"])
            self.assertEqual(docker_rm_args[:3], ["docker", "rm", "-f"])

    @patch.object(ToolRegistry, "_ensure_docker_container", autospec=True)
    def test_docker_workspace_scripts_handle_list_and_edit_operations(self, mock_ensure_docker_container) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir) / "workspace"
            workspace.mkdir()
            (workspace / "index.html").write_text("<div>before</div>\n", encoding="utf-8")
            (workspace / "README.md").write_text("hello\nworld\n", encoding="utf-8")

            registry = ToolRegistry(
                workspace,
                config={
                    "workspace": {
                        "command_runner": {
                            "mode": "docker",
                            "tool_scope": "all",
                        }
                    }
                },
            )

            def run_script_locally(script: str, args=None, *, input_text=None):
                completed = subprocess.run(
                    [sys.executable, "-c", script, *(args or [])],
                    cwd=workspace,
                    input=input_text,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                output = completed.stdout
                if completed.stderr:
                    output += f"\nSTDERR:\n{completed.stderr}"
                if completed.returncode != 0:
                    raise RuntimeError(output or f"Command failed with exit code {completed.returncode}")
                return output

            registry._run_python_in_docker = run_script_locally  # type: ignore[method-assign]

            expected_listing = [
                f"{item.name} ({item.stat().st_size} bytes)"
                for item in sorted(workspace.iterdir())
            ]
            listing = registry.list_directory()
            chunk = registry.read_file("README.md", offset=2, limit=1)
            edit_result = registry.edit_file("index.html", "before", "after")

            self.assertEqual(listing, expected_listing)
            self.assertEqual(chunk, "world\n")
            self.assertEqual(edit_result, "Successfully edited index.html")
            self.assertEqual((workspace / "index.html").read_text(encoding="utf-8"), "<div>after</div>\n")
            mock_ensure_docker_container.assert_not_called()

    def test_run_manifest_tracks_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "run_manifest.json"
            manifest = RunManifest(path, run_id="test-run")
            prompts = ["alpha", "beta"]
            manifest.seed_prompts(prompts, {"alpha"})
            prompt_id = manifest.make_prompt_id(1, "beta")
            manifest.mark_running(prompt_id, 1, "beta", Path(tmp_dir) / "sandbox/session_000001")
            manifest.mark_result(
                prompt_id,
                status="retryable_error",
                completed=False,
                retryable=True,
                error="rate limited",
                turns=3,
                tool_calls_count=1,
                usage={"total_tokens": 10},
                workspace_dir=Path(tmp_dir) / "sandbox/session_000001",
            )
            manifest.set_route(prompt_id, "error_dataset")
            snapshot = manifest.snapshot()
            self.assertEqual(snapshot["summary"]["total_prompts"], 2)
            self.assertEqual(snapshot["summary"]["status_counts"]["completed"], 1)
            self.assertEqual(snapshot["entries"][prompt_id]["dataset_route"], "error_dataset")
            self.assertEqual(snapshot["entries"][prompt_id]["retryable"], True)

    def test_route_entry_marks_max_turns_exceeded_as_error_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_path = Path(tmp_dir) / "dataset.jsonl"
            manifest_path = Path(tmp_dir) / "run_manifest.json"
            generator = AgenticDatasetGenerator.__new__(AgenticDatasetGenerator)
            generator.formatter = Formatter()
            generator.output_file = dataset_path
            generator.write_lock = threading.Lock()
            generator.run_id = "run_test"
            generator.run_manifest = RunManifest(manifest_path, run_id="run_test")

            prompt = "Build a dashboard"
            generator.run_manifest.seed_prompts([prompt], set())
            prompt_id = generator.run_manifest.make_prompt_id(0, prompt)

            entry = {
                "prompt": prompt,
                "messages": [
                    {"role": "system", "content": "You are a coding agent."},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": "I am still working on it."},
                ],
                "metadata": {
                    "prompt_id": prompt_id,
                    "run_id": "run_test",
                    "session_id": "session_000000",
                    "turns": 50,
                    "completed": False,
                    "tool_calls_count": 0,
                    "error": "LLM call failed: max turns exceeded",
                    "retryable": True,
                },
                "usage": {"total_tokens": 12, "cost": 0.0},
            }

            generator._route_entry(entry)

            snapshot = generator.run_manifest.snapshot()
            self.assertFalse(dataset_path.exists())
            self.assertEqual(snapshot["entries"][prompt_id]["dataset_route"], "error_dataset")

    @patch("agentic_datagen.generator.create_session_engine")
    def test_process_prompt_retries_retryable_session_error_in_same_run(self, mock_create_session_engine) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            sandbox = Path(tmp_dir) / "sandbox"
            sandbox.mkdir()
            manifest_path = Path(tmp_dir) / "run_manifest.json"
            prompt = "Build a dashboard"

            generator = AgenticDatasetGenerator.__new__(AgenticDatasetGenerator)
            generator.config = normalize_config(
                {
                    "workspace": {
                        "base_dir": str(sandbox),
                        "cleanup": False,
                        "preserve_on_error": True,
                    },
                    "processing": {"retryable_session_max_attempts": 2},
                    "tools": {"enabled": []},
                }
            )
            generator.logger = unittest.mock.Mock()
            generator.formatter = Formatter()
            generator.run_id = "run_test"
            generator.base_workspace_dir = sandbox
            generator.tool_definitions = []
            generator.run_manifest = RunManifest(manifest_path, run_id="run_test")

            first_session = unittest.mock.Mock()
            first_session.run.return_value = {
                "session_id": "session_000000",
                "prompt_id": generator.run_manifest.make_prompt_id(0, prompt),
                "run_id": "run_test",
                "prompt": prompt,
                "turns": 1,
                "conversation": [
                    {"role": "system", "content": "You are a coding agent."},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": ""},
                ],
                "tool_calls": [],
                "final_response": "",
                "completed": False,
                "error": "LLM call failed: empty final response",
                "retryable": True,
                "usage": {"total_tokens": 3, "cost": 0.0},
            }
            second_session = unittest.mock.Mock()
            second_session.run.return_value = {
                "session_id": "session_000000",
                "prompt_id": generator.run_manifest.make_prompt_id(0, prompt),
                "run_id": "run_test",
                "prompt": prompt,
                "turns": 2,
                "conversation": [
                    {"role": "system", "content": "You are a coding agent."},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": "Done."},
                ],
                "tool_calls": [],
                "final_response": "Done.",
                "completed": True,
                "error": None,
                "retryable": False,
                "usage": {"total_tokens": 6, "cost": 0.0},
            }
            mock_create_session_engine.side_effect = [first_session, second_session]

            generator.run_manifest.seed_prompts([prompt], set())

            entry = generator._process_prompt(prompt, 0)

            self.assertIsNotNone(entry)
            self.assertEqual(mock_create_session_engine.call_count, 2)
            snapshot = generator.run_manifest.snapshot()
            prompt_id = generator.run_manifest.make_prompt_id(0, prompt)
            self.assertEqual(snapshot["entries"][prompt_id]["attempt_count"], 2)
            self.assertEqual(snapshot["entries"][prompt_id]["status"], "completed")
            self.assertIsNone(snapshot["entries"][prompt_id]["last_error"])

    @patch("agentic_datagen.generator.create_session_engine")
    def test_process_prompt_logs_terminal_retryable_error_once_after_retries_exhausted(self, mock_create_session_engine) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            sandbox = Path(tmp_dir) / "sandbox"
            sandbox.mkdir()
            manifest_path = Path(tmp_dir) / "run_manifest.json"
            prompt = "Build a dashboard"

            generator = AgenticDatasetGenerator.__new__(AgenticDatasetGenerator)
            generator.config = normalize_config(
                {
                    "workspace": {
                        "base_dir": str(sandbox),
                        "cleanup": False,
                        "preserve_on_error": True,
                    },
                    "processing": {"retryable_session_max_attempts": 2},
                    "tools": {"enabled": []},
                }
            )
            generator.logger = unittest.mock.Mock()
            generator.formatter = Formatter()
            generator.run_id = "run_test"
            generator.base_workspace_dir = sandbox
            generator.tool_definitions = []
            generator.run_manifest = RunManifest(manifest_path, run_id="run_test")

            retryable_error_session = unittest.mock.Mock()
            retryable_error_session.run.return_value = {
                "session_id": "session_000000",
                "prompt_id": generator.run_manifest.make_prompt_id(0, prompt),
                "run_id": "run_test",
                "prompt": prompt,
                "turns": 1,
                "conversation": [
                    {"role": "system", "content": "You are a coding agent."},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": ""},
                ],
                "tool_calls": [],
                "final_response": "",
                "completed": False,
                "error": "LLM call failed: empty final response",
                "retryable": True,
                "usage": {"total_tokens": 3, "cost": 0.0},
            }
            mock_create_session_engine.side_effect = [retryable_error_session, retryable_error_session]

            generator.run_manifest.seed_prompts([prompt], set())

            entry = generator._process_prompt(prompt, 0)

            self.assertIsNotNone(entry)
            self.assertEqual(mock_create_session_engine.call_count, 2)
            preserve_logs = [
                call
                for call in generator.logger.info.call_args_list
                if call.args and isinstance(call.args[0], str) and call.args[0].startswith("Preserving workspace:")
            ]
            self.assertEqual(len(preserve_logs), 1)
            generator.logger.error.assert_any_call("Session error: LLM call failed: empty final response")

    def test_run_manifest_resets_cleaned_out_completed_prompt_to_pending(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "run_manifest.json"
            manifest = RunManifest(path, run_id="test-run")
            prompts = ["alpha", "beta"]
            manifest.seed_prompts(prompts, {"alpha", "beta"})
            manifest.seed_prompts(prompts, {"alpha"})

            beta_prompt_id = manifest.make_prompt_id(1, "beta")
            snapshot = manifest.snapshot()

            self.assertEqual(snapshot["entries"][beta_prompt_id]["status"], "pending")
            self.assertEqual(snapshot["entries"][beta_prompt_id]["completed"], False)
            self.assertEqual(snapshot["entries"][beta_prompt_id]["retryable"], True)
            self.assertEqual(
                snapshot["entries"][beta_prompt_id]["last_error"],
                "Removed from dataset; queued for retry",
            )

    def test_custom_python_tool_registration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir) / "workspace"
            workspace.mkdir()
            (workspace / "hello.txt").write_text("hello", encoding="utf-8")
            registry = ToolRegistry(
                workspace,
                config={"tools": {"custom_python_modules": ["custom_tools.example_tools"]}},
            )
            definitions = registry.get_tool_definitions(["workspace_snapshot"])
            self.assertEqual(len(definitions), 1)
            result = registry.execute_tool("workspace_snapshot", {"limit": 5})
            self.assertEqual(result["success"], True)
            items = result["result"]["items"]
            self.assertTrue(any(item["path"] == "hello.txt" for item in items))

    def test_mcp_http_tool_registration_and_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir) / "workspace"
            workspace.mkdir()
            server = ThreadingHTTPServer(("127.0.0.1", 0), MCPHandler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                url = f"http://127.0.0.1:{server.server_port}"
                registry = ToolRegistry(
                    workspace,
                    config={
                        "tools": {
                            "mcp_servers": {
                                "demo": {
                                    "transport": "http",
                                    "url": url,
                                    "tool_name_prefix": "demo",
                                }
                            }
                        }
                    },
                )
                tool_name = "demo__echo"
                definitions = registry.get_tool_definitions([tool_name])
                self.assertEqual(len(definitions), 1)
                result = registry.execute_tool(tool_name, {"text": "hello"})
                self.assertEqual(result["success"], True)
                self.assertEqual(result["result"]["text"], "echo:hello")

                wildcard_definitions = registry.get_tool_definitions(["demo:*"])
                self.assertEqual(len(wildcard_definitions), 1)
                self.assertEqual(
                    wildcard_definitions[0]["function"]["name"],
                    "demo__echo",
                )
            finally:
                server.shutdown()
                server.server_close()


if __name__ == "__main__":
    unittest.main()

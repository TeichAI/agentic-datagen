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
from agentic_datagen.generator import build_dataset_readme
from agentic_datagen.run_manifest import RunManifest
from agentic_datagen.tool_registry import ToolRegistry
from agentic_datagen.utils import load_prompts


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

        self.assertEqual(canonical["messages"][2]["thinking"], "I should create the file first.")
        self.assertEqual(canonical["messages"][2]["content"], "")
        self.assertEqual(
            canonical["messages"][2]["tool_calls"][0]["function"]["arguments"],
            {"path": "index.html", "content": "<h1>Hello</h1>"},
        )
        self.assertEqual(canonical["messages"][4]["thinking"], "The file is ready.")
        self.assertEqual(canonical["messages"][4]["content"], "Done.")

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
        self.assertIn('path: "agentic_dataset.jsonl"', readme)
        self.assertIn('"thinking": "..."', readme)
        self.assertIn('"tool_calls"', readme)
        self.assertIn('"tools"', readme)
        self.assertIn("- Reasoning effort: `high`", readme)

    def test_read_file_supports_offset_and_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir) / "workspace"
            workspace.mkdir()
            (workspace / "notes.txt").write_text("alpha\nbeta\ngamma\ndelta\n", encoding="utf-8")
            registry = ToolRegistry(workspace)

            self.assertEqual(registry.read_file("notes.txt", offset=2, limit=2), "beta\ngamma\n")
            self.assertEqual(registry.read_file("notes.txt", offset=3), "gamma\ndelta\n")
            self.assertEqual(registry.read_file("notes.txt", limit=1), "alpha\n")

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
            registry = ToolRegistry(workspace)

            listing = registry.list_directory(file_path="nested")
            output = registry.run_command("printf hello", timeout=1)

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
    def test_workspace_tools_use_docker_when_full_scope_configured(self, mock_run) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace = Path(tmp_dir) / "workspace"
            workspace.mkdir()
            mock_run.side_effect = [
                subprocess.CompletedProcess(args=["docker", "run"], returncode=0, stdout="container-id\n", stderr=""),
                subprocess.CompletedProcess(args=["docker", "inspect"], returncode=0, stdout="true\n", stderr=""),
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
            docker_write_args = mock_run.call_args_list[2].args[0]
            docker_read_args = mock_run.call_args_list[4].args[0]
            docker_list_args = mock_run.call_args_list[6].args[0]
            docker_rm_args = mock_run.call_args_list[7].args[0]
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
                    ["python3", "-c", script, *(args or [])],
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

            listing = registry.list_directory()
            chunk = registry.read_file("README.md", offset=2, limit=1)
            edit_result = registry.edit_file("index.html", "before", "after")

            self.assertEqual(listing, ["README.md (12 bytes)", "index.html (18 bytes)"])
            self.assertEqual(chunk, "world\n")
            self.assertEqual(edit_result, "Successfully edited index.html")
            self.assertEqual((workspace / "index.html").read_text(encoding="utf-8"), "<div>after</div>\n")
            mock_ensure_docker_container.assert_called_once()

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

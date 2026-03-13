from __future__ import annotations

import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from run_manifest import RunManifest
from tool_registry import ToolRegistry


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

from __future__ import annotations

import importlib
import inspect
import json
import os
import shlex
import subprocess
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import requests


@dataclass
class RegisteredTool:
    name: str
    description: str
    parameters: Dict[str, Any]
    handler: Callable[..., Any]
    source: str = "builtin"
    title: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_definition(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class MCPHTTPToolClient:
    def __init__(
        self,
        server_name: str,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        timeout: float = 30.0,
    ):
        self.server_name = server_name
        self.url = url
        self.timeout = timeout
        self.headers = {"Content-Type": "application/json", **(headers or {})}
        self._session = requests.Session()
        self._lock = threading.Lock()
        self._request_id = 0
        self._initialized = False

    def _next_id(self) -> int:
        with self._lock:
            self._request_id += 1
            return self._request_id

    def _rpc(self, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": method,
        }
        if params is not None:
            payload["params"] = params
        response = self._session.post(
            self.url,
            headers=self.headers,
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        body = response.json()
        if "error" in body:
            raise RuntimeError(json.dumps(body["error"], ensure_ascii=False))
        return body.get("result") or {}

    def _notify_initialized(self) -> None:
        payload = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
        }
        response = self._session.post(
            self.url,
            headers=self.headers,
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()

    def ensure_initialized(self) -> None:
        if self._initialized:
            return
        try:
            self._rpc(
                "initialize",
                {
                    "protocolVersion": "2025-03-26",
                    "capabilities": {},
                    "clientInfo": {
                        "name": "agentic-datagen",
                        "version": "1.0",
                    },
                },
            )
            self._notify_initialized()
        except Exception:
            pass
        self._initialized = True

    def list_tools(self) -> List[Dict[str, Any]]:
        self.ensure_initialized()
        tools: List[Dict[str, Any]] = []
        cursor: Optional[str] = None
        while True:
            params = {"cursor": cursor} if cursor else {}
            result = self._rpc("tools/list", params)
            tools.extend(result.get("tools") or [])
            cursor = result.get("nextCursor")
            if not cursor:
                break
        return tools

    def call_tool(self, remote_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        self.ensure_initialized()
        result = self._rpc(
            "tools/call",
            {
                "name": remote_name,
                "arguments": arguments,
            },
        )
        content = result.get("content") or []
        text_parts: List[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text")
                if text:
                    text_parts.append(text)
        return {
            "content": content,
            "text": "\n\n".join(text_parts) if text_parts else None,
            "is_error": bool(result.get("isError")),
            "server": self.server_name,
            "remote_tool": remote_name,
        }

    def close(self) -> None:
        self._session.close()


class ToolRegistry:
    def __init__(self, workspace_dir: Path, config: Optional[Dict[str, Any]] = None):
        self.workspace_dir = workspace_dir
        self.config = config or {}
        self._tools: Dict[str, RegisteredTool] = {}
        self._mcp_clients: Dict[str, MCPHTTPToolClient] = {}
        self._docker_container_name: Optional[str] = None
        self._docker_container_running = False
        self._docker_bootstrap_completed = False
        self._register_builtin_tools()
        self._register_custom_python_tools()
        self._register_mcp_tools()
        if self._command_runner_mode() == "docker" and self._docker_eager_start():
            self._ensure_docker_container()

    def register_tool(
        self,
        spec: Dict[str, Any],
        *,
        source: str = "custom",
    ) -> None:
        name = spec.get("name")
        description = spec.get("description")
        parameters = spec.get("parameters") or {
            "type": "object",
            "properties": {},
            "required": [],
        }
        handler = spec.get("handler")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Tool spec missing valid 'name'")
        if not isinstance(description, str) or not description.strip():
            raise ValueError(f"Tool {name} missing valid 'description'")
        if not isinstance(parameters, dict):
            raise ValueError(f"Tool {name} has invalid 'parameters'")
        if not callable(handler):
            raise ValueError(f"Tool {name} missing callable 'handler'")
        self._tools[name] = RegisteredTool(
            name=name,
            description=description,
            parameters=parameters,
            handler=handler,
            source=spec.get("source", source),
            title=spec.get("title"),
            metadata=spec.get("metadata") or {},
        )

    def available_tools(self) -> List[RegisteredTool]:
        return list(self._tools.values())

    def get_tool_definitions(self, enabled_tools: List[str]) -> List[Dict[str, Any]]:
        definitions: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for tool_name in enabled_tools:
            if tool_name.endswith(":*"):
                selector = tool_name[:-2]
                for tool in self.available_tools():
                    if tool.name in seen:
                        continue
                    if not self._matches_selector(tool, selector):
                        continue
                    definitions.append(tool.to_definition())
                    seen.add(tool.name)
                continue
            if tool_name in seen:
                continue
            tool = self._tools.get(tool_name)
            if tool is None:
                continue
            definitions.append(tool.to_definition())
            seen.add(tool_name)
        return definitions

    def _matches_selector(self, tool: RegisteredTool, selector: str) -> bool:
        normalized = selector.strip()
        if not normalized:
            return False
        return (
            tool.source == normalized
            or tool.source == f"mcp:{normalized}"
            or tool.source == f"python:{normalized}"
            or tool.name.startswith(f"{normalized}__")
        )

    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        tool = self._tools.get(tool_name)
        if tool is None:
            return {"success": False, "error": f"Unknown tool: {tool_name}"}
        try:
            result = self._invoke_handler(tool.handler, arguments)
            return {
                "success": True,
                "result": result,
                "tool": tool_name,
                "source": tool.source,
            }
        except Exception as exc:
            return {
                "success": False,
                "error": str(exc),
                "tool": tool_name,
                "source": tool.source,
            }

    def _invoke_handler(self, handler: Callable[..., Any], arguments: Dict[str, Any]) -> Any:
        signature = inspect.signature(handler)
        kwargs = dict(arguments)
        if "context" in signature.parameters and "context" not in kwargs:
            kwargs["context"] = self._build_context()
        if "workspace_dir" in signature.parameters and "workspace_dir" not in kwargs:
            kwargs["workspace_dir"] = self.workspace_dir
        if "config" in signature.parameters and "config" not in kwargs:
            kwargs["config"] = self.config
        if "registry" in signature.parameters and "registry" not in kwargs:
            kwargs["registry"] = self
        return handler(**kwargs)

    def _build_context(self) -> Dict[str, Any]:
        return {
            "workspace_dir": self.workspace_dir,
            "config": self.config,
            "registry": self,
        }

    def _command_runner_config(self) -> Dict[str, Any]:
        workspace_config = self.config.get("workspace") or {}
        command_runner = workspace_config.get("command_runner") or {}
        return command_runner if isinstance(command_runner, dict) else {}

    def _command_timeout_seconds(self) -> float:
        command_runner = self._command_runner_config()
        timeout = command_runner.get("timeout_seconds", 30)
        try:
            return float(timeout)
        except (TypeError, ValueError):
            return 30.0

    def _command_runner_mode(self) -> str:
        mode = self._command_runner_config().get("mode", "docker")
        return str(mode).strip().lower() or "docker"

    def _command_runner_tool_scope(self) -> str:
        scope = self._command_runner_config().get("tool_scope", "all")
        normalized = str(scope).strip().lower() or "all"
        if normalized in {"all", "full", "workspace"}:
            return "all"
        return "command"

    def _docker_eager_start(self) -> bool:
        return bool(self._command_runner_config().get("eager_start", False))

    def _run_workspace_tools_in_docker(self) -> bool:
        return self._command_runner_mode() == "docker" and self._command_runner_tool_scope() == "all"

    def _docker_bootstrap_commands(self) -> List[str]:
        commands = self._command_runner_config().get("bootstrap_commands") or []
        if not isinstance(commands, list):
            return []
        return [str(command) for command in commands if str(command).strip()]

    def _docker_bootstrap_trigger(self) -> str:
        trigger = self._command_runner_config().get("bootstrap_trigger", "before_first_command")
        normalized = str(trigger).strip().lower() or "before_first_command"
        if normalized in {"start", "container_start", "on_start"}:
            return "container_start"
        return "before_first_command"

    def _docker_bootstrap_timeout_seconds(self) -> float:
        timeout = self._command_runner_config().get(
            "bootstrap_timeout_seconds",
            self._command_timeout_seconds(),
        )
        try:
            return float(timeout)
        except (TypeError, ValueError):
            return self._command_timeout_seconds()

    def _docker_binary(self) -> str:
        binary = self._command_runner_config().get("docker_binary", "docker")
        return str(binary).strip() or "docker"

    def _docker_container_workspace_dir(self) -> str:
        path = self._command_runner_config().get("container_workspace_dir", "/workspace")
        return str(path).strip() or "/workspace"

    def _docker_shell(self) -> str:
        shell = self._command_runner_config().get("shell", "/bin/sh")
        return str(shell).strip() or "/bin/sh"

    def _docker_shell_args(self) -> List[str]:
        shell_args = self._command_runner_config().get("shell_args")
        if isinstance(shell_args, list) and shell_args:
            return [str(item) for item in shell_args]
        return ["-lc"]

    def _docker_image(self) -> str:
        image = self._command_runner_config().get("docker_image") or self._command_runner_config().get("image")
        if not image:
            image = "agentic-datagen-session-runtime:latest"
        return str(image)

    def _docker_container_name_for_workspace(self) -> str:
        if self._docker_container_name:
            return self._docker_container_name
        safe_workspace = "".join(
            char if char.isalnum() else "-" for char in self.workspace_dir.name.lower()
        ).strip("-") or "workspace"
        self._docker_container_name = f"agentic-datagen-{safe_workspace}-{uuid.uuid4().hex[:8]}"
        return self._docker_container_name

    def _run_subprocess(
        self,
        args: List[str],
        *,
        timeout: Optional[float] = None,
        cwd: Optional[Path] = None,
        input_text: Optional[str] = None,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
            input=input_text,
        )

    def _docker_environment_args(self) -> List[str]:
        environment = dict(self._command_runner_config().get("environment") or {})
        if not isinstance(environment, dict):
            return []
        if "HOME" not in environment:
            environment["HOME"] = f"{self._docker_container_workspace_dir()}/.agent-home"
        args: List[str] = []
        for key, value in environment.items():
            args.extend(["-e", f"{key}={value}"])
        return args

    def _docker_user(self) -> Optional[str]:
        if not bool(self._command_runner_config().get("use_host_user", True)):
            return None
        if not hasattr(os, "getuid") or not hasattr(os, "getgid"):
            return None
        return f"{os.getuid()}:{os.getgid()}"

    def _docker_user_args(self) -> List[str]:
        docker_user = self._docker_user()
        if not docker_user:
            return []
        return ["--user", docker_user]

    def _docker_create_args(self) -> List[str]:
        create_args = self._command_runner_config().get("create_args") or []
        if not isinstance(create_args, list):
            return []
        return [str(item) for item in create_args]

    def _docker_exec_args(self) -> List[str]:
        exec_args = self._command_runner_config().get("exec_args") or []
        if not isinstance(exec_args, list):
            return []
        return [str(item) for item in exec_args]

    def _docker_container_is_running(self) -> bool:
        if not self._docker_container_name or not self._docker_container_running:
            return False
        result = self._run_subprocess(
            [
                self._docker_binary(),
                "inspect",
                "-f",
                "{{.State.Running}}",
                self._docker_container_name,
            ],
            timeout=10,
        )
        return result.returncode == 0 and result.stdout.strip() == "true"

    def _docker_start_error_message(self, result: subprocess.CompletedProcess[str]) -> str:
        output = (result.stderr or result.stdout or "docker run failed").strip()
        image = self._docker_image()
        lowered = output.lower()
        missing_image_tokens = [
            "unable to find image",
            "pull access denied",
            "repository does not exist",
            "manifest unknown",
            "no such image",
        ]
        if any(token in lowered for token in missing_image_tokens):
            return (
                f"Docker image '{image}' is not available. "
                "Build or pull it before running sessions. "
                "For the bundled default runtime, run "
                "`docker build -t agentic-datagen-session-runtime:latest -f src/docker/session-runtime.Dockerfile .`.\n"
                f"{output}"
            )
        return output

    def validate_runtime_prerequisites(self) -> None:
        if self._command_runner_mode() != "docker":
            return
        image = self._docker_image()
        result = self._run_subprocess(
            [self._docker_binary(), "image", "inspect", image],
            timeout=10,
        )
        if result.returncode != 0:
            raise RuntimeError(self._docker_start_error_message(result))

    def _ensure_docker_container(self) -> None:
        if self._docker_container_is_running():
            return
        container_name = self._docker_container_name_for_workspace()
        container_workspace_dir = self._docker_container_workspace_dir()
        result = self._run_subprocess(
            [
                self._docker_binary(),
                "run",
                "-d",
                "--rm",
                "--name",
                container_name,
                "-w",
                container_workspace_dir,
                "-v",
                f"{self.workspace_dir.resolve()}:{container_workspace_dir}",
                *self._docker_user_args(),
                *self._docker_environment_args(),
                *self._docker_create_args(),
                self._docker_image(),
                "tail",
                "-f",
                "/dev/null",
            ],
            timeout=self._command_timeout_seconds(),
        )
        if result.returncode != 0:
            raise RuntimeError(self._docker_start_error_message(result))
        self._docker_container_running = True
        self._docker_bootstrap_completed = False
        self._maybe_run_docker_bootstrap("container_start")

    def _docker_exec_command(
        self,
        command: str,
        *,
        input_text: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> str:
        result = self._run_subprocess(
            [
                self._docker_binary(),
                "exec",
                "-i",
                "-w",
                self._docker_container_workspace_dir(),
                *self._docker_user_args(),
                *self._docker_exec_args(),
                self._docker_container_name_for_workspace(),
                self._docker_shell(),
                *self._docker_shell_args(),
                command,
            ],
            timeout=timeout or self._command_timeout_seconds(),
            input_text=input_text,
        )
        output = result.stdout
        if result.stderr:
            output += f"\nSTDERR:\n{result.stderr}"
        if result.returncode != 0:
            raise RuntimeError(
                output or f"Command failed with exit code {result.returncode}"
            )
        return output or "Command executed successfully (no output)"

    def _maybe_run_docker_bootstrap(self, trigger: str) -> None:
        if self._docker_bootstrap_completed:
            return
        commands = self._docker_bootstrap_commands()
        if not commands:
            self._docker_bootstrap_completed = True
            return
        if self._docker_bootstrap_trigger() != trigger:
            return
        timeout = self._docker_bootstrap_timeout_seconds()
        for index, command in enumerate(commands, 1):
            try:
                self._docker_exec_command(command, timeout=timeout)
            except Exception as exc:
                raise RuntimeError(
                    f"Docker bootstrap command {index} failed: {command}\n{exc}"
                ) from exc
        self._docker_bootstrap_completed = True

    def _run_command_in_docker(self, command: str) -> str:
        self._ensure_docker_container()
        self._maybe_run_docker_bootstrap("before_first_command")
        return self._docker_exec_command(command, timeout=self._command_timeout_seconds())

    def _resolve_workspace_path(self, path_value: str = "") -> Path:
        workspace_root = self.workspace_dir.resolve()
        normalized = str(path_value or "").strip()
        if not normalized or normalized == ".":
            return workspace_root

        candidate_path = Path(normalized)
        if candidate_path.is_absolute():
            container_root = Path(self._docker_container_workspace_dir())
            try:
                relative = candidate_path.relative_to(container_root)
                candidate = (workspace_root / relative).resolve()
            except ValueError:
                candidate = candidate_path.resolve()
        else:
            candidate = (workspace_root / candidate_path).resolve()

        if not str(candidate).startswith(str(workspace_root)):
            raise PermissionError("Access denied: path outside workspace")
        return candidate

    def _workspace_relative_path(self, relative_path: str = "") -> str:
        workspace_root = self.workspace_dir.resolve()
        candidate = self._resolve_workspace_path(relative_path)
        if candidate == workspace_root:
            return "."
        return candidate.relative_to(workspace_root).as_posix()

    def _run_python_in_docker(
        self,
        script: str,
        args: Optional[List[str]] = None,
        *,
        input_text: Optional[str] = None,
    ) -> str:
        command = " ".join(
            [
                "python3",
                "-c",
                shlex.quote(script),
                *[shlex.quote(item) for item in (args or [])],
            ]
        )
        self._ensure_docker_container()
        return self._docker_exec_command(
            command,
            timeout=self._command_timeout_seconds(),
            input_text=input_text,
        )

    def _stop_docker_container(self) -> None:
        if not self._docker_container_name or not self._docker_container_running:
            return
        self._run_subprocess(
            [self._docker_binary(), "rm", "-f", self._docker_container_name],
            timeout=10,
        )
        self._docker_container_running = False
        self._docker_bootstrap_completed = False

    def _register_builtin_tools(self) -> None:
        builtin_specs = [
            {
                "name": "read_file",
                "description": "Read a UTF-8 text file from the workspace before editing or to inspect generated code. Use this before edit_file so replacements are based on the exact current file contents. Optionally pass offset and limit to read a specific line range.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the file relative to the workspace root.",
                        },
                        "offset": {
                            "type": "integer",
                            "description": "Optional 1-indexed starting line number.",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Optional number of lines to read starting from offset.",
                        }
                    },
                    "required": ["file_path"],
                },
                "handler": self.read_file,
            },
            {
                "name": "write_file",
                "description": "Create a new file or fully overwrite an existing workspace file with the provided UTF-8 content. Parent directories are created automatically. Prefer this for new files or full rewrites rather than using edit_file for large structural changes.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the file relative to the workspace root.",
                        },
                        "content": {
                            "type": "string",
                            "description": "Full file contents to write.",
                        },
                    },
                    "required": ["file_path", "content"],
                },
                "handler": self.write_file,
            },
            {
                "name": "edit_file",
                "description": "Replace the first occurrence of old_text with new_text in an existing workspace file. Read the file immediately beforehand and copy the exact text to replace, because edit_file fails if old_text does not match the current file contents exactly.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the file relative to the workspace root.",
                        },
                        "old_text": {
                            "type": "string",
                            "description": "Exact text to replace.",
                        },
                        "new_text": {
                            "type": "string",
                            "description": "Replacement text.",
                        },
                    },
                    "required": ["file_path", "old_text", "new_text"],
                },
                "handler": self.edit_file,
            },
            {
                "name": "list_directory",
                "description": "List files and directories in the workspace or a relative subdirectory to understand project structure before reading, editing, or running commands.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "dir_path": {
                            "type": "string",
                            "description": "Optional relative directory path. Leave empty to list the workspace root.",
                        }
                    },
                    "required": [],
                },
                "handler": self.list_directory,
            },
            {
                "name": "search_code",
                "description": "Case-insensitive substring search across workspace files, returning file, line number, and matched content. Use this to locate anchors before calling read_file or edit_file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": "Text pattern to search for.",
                        },
                        "file_pattern": {
                            "type": "string",
                            "description": "Optional glob-like filename filter such as '*.py' or '*.ts'.",
                        },
                    },
                    "required": ["pattern"],
                },
                "handler": self.search_code,
            },
            {
                "name": "run_command",
                "description": "Execute a shell command inside the workspace and return stdout plus stderr. Use this for installs, builds, tests, linters, and starting dev servers after the necessary files are in place.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "Shell command to execute inside the workspace directory.",
                        },
                        "timeout": {
                            "type": "number",
                            "description": "Optional timeout in seconds for this command invocation.",
                        }
                    },
                    "required": ["command"],
                },
                "handler": self.run_command,
            },
            {
                "name": "web_search",
                "description": "Search the web through the configured SearXNG instance and return the top results as text snippets.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query to send to the configured SearXNG instance.",
                        }
                    },
                    "required": ["query"],
                },
                "handler": self.web_search,
            },
        ]
        for spec in builtin_specs:
            self.register_tool(spec, source="builtin")

    def _register_custom_python_tools(self) -> None:
        tools_config = self.config.get("tools") or {}
        module_names = tools_config.get("custom_python_modules") or []
        for module_name in module_names:
            module = importlib.import_module(module_name)
            if hasattr(module, "TOOLS"):
                for spec in getattr(module, "TOOLS"):
                    self.register_tool(spec, source=f"python:{module_name}")
            if hasattr(module, "register_tools"):
                returned = module.register_tools(self)
                if returned:
                    for spec in returned:
                        self.register_tool(spec, source=f"python:{module_name}")

    def _register_mcp_tools(self) -> None:
        tools_config = self.config.get("tools") or {}
        server_configs = tools_config.get("mcp_servers") or {}
        strict_mode = bool(tools_config.get("strict_mcp", False))
        if isinstance(server_configs, list):
            iterable = [(f"server_{index}", item) for index, item in enumerate(server_configs)]
        else:
            iterable = list(server_configs.items())
        for server_name, server_config in iterable:
            if not isinstance(server_config, dict):
                continue
            transport = server_config.get("transport", "http")
            if transport != "http":
                continue
            url = server_config.get("url")
            if not url:
                continue
            try:
                client = MCPHTTPToolClient(
                    server_name=server_name,
                    url=url,
                    headers=server_config.get("headers"),
                    timeout=float(server_config.get("timeout", 30.0)),
                )
                self._mcp_clients[server_name] = client
                prefix = server_config.get("tool_name_prefix")
                for remote_tool in client.list_tools():
                    remote_name = remote_tool.get("name")
                    if not remote_name:
                        continue
                    local_name = (
                        f"{prefix}__{remote_name}"
                        if prefix
                        else f"mcp__{server_name}__{remote_name}"
                    )
                    description = remote_tool.get("description") or f"Remote MCP tool '{remote_name}' from server '{server_name}'."
                    input_schema = remote_tool.get("inputSchema") or {
                        "type": "object",
                        "properties": {},
                        "required": [],
                    }

                    def handler_factory(
                        bound_client: MCPHTTPToolClient,
                        bound_remote_name: str,
                    ) -> Callable[..., Dict[str, Any]]:
                        def _handler(**kwargs: Any) -> Dict[str, Any]:
                            return bound_client.call_tool(bound_remote_name, kwargs)

                        return _handler

                    self.register_tool(
                        {
                            "name": local_name,
                            "title": remote_tool.get("title"),
                            "description": f"[MCP:{server_name}] {description}",
                            "parameters": input_schema,
                            "handler": handler_factory(client, remote_name),
                            "metadata": {
                                "remote_name": remote_name,
                                "server": server_name,
                            },
                        },
                        source=f"mcp:{server_name}",
                    )
            except Exception:
                if strict_mode:
                    raise

    def _ensure_in_workspace(self, path: Path) -> None:
        if not str(path.resolve()).startswith(str(self.workspace_dir.resolve())):
            raise PermissionError("Access denied: path outside workspace")

    def _slice_file_content(
        self,
        content: str,
        offset: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> str:
        if offset is None and limit is None:
            return content
        start = max((offset or 1) - 1, 0)
        end = None if limit is None else max(start + limit, start)
        return "".join(content.splitlines(keepends=True)[start:end])

    def read_file(
        self,
        file_path: str,
        offset: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> str:
        if self._run_workspace_tools_in_docker():
            script = (
                "from pathlib import Path\n"
                "import sys\n"
                "path = Path(sys.argv[1])\n"
                "offset = None if sys.argv[2] == '__NONE__' else max(int(sys.argv[2]), 1)\n"
                "limit = None if sys.argv[3] == '__NONE__' else max(int(sys.argv[3]), 0)\n"
                "content = path.read_text(encoding='utf-8')\n"
                "if offset is None and limit is None:\n"
                "    print(content, end='')\n"
                "else:\n"
                "    lines = content.splitlines(keepends=True)\n"
                "    start = max((offset or 1) - 1, 0)\n"
                "    end = None if limit is None else max(start + limit, start)\n"
                "    print(''.join(lines[start:end]), end='')\n"
            )
            return self._run_python_in_docker(
                script,
                [
                    self._workspace_relative_path(file_path),
                    "__NONE__" if offset is None else str(offset),
                    "__NONE__" if limit is None else str(limit),
                ],
            )
        full_path = self._resolve_workspace_path(file_path)
        if not full_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        self._ensure_in_workspace(full_path)
        return self._slice_file_content(
            full_path.read_text(encoding="utf-8"),
            offset=offset,
            limit=limit,
        )

    def write_file(self, file_path: str, content: str) -> str:
        if self._run_workspace_tools_in_docker():
            script = (
                "from pathlib import Path; import sys; "
                "path = Path(sys.argv[1]); "
                "path.parent.mkdir(parents=True, exist_ok=True); "
                "payload = sys.stdin.read(); "
                "path.write_text(payload, encoding='utf-8'); "
                "print(f'Successfully wrote {len(payload)} characters to {sys.argv[1]}', end='')"
            )
            return self._run_python_in_docker(
                script,
                [self._workspace_relative_path(file_path)],
                input_text=content,
            )
        full_path = self._resolve_workspace_path(file_path)
        self._ensure_in_workspace(full_path)
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content, encoding="utf-8")
        return f"Successfully wrote {len(content)} characters to {file_path}"

    def edit_file(self, file_path: str, old_text: str, new_text: str) -> str:
        if self._run_workspace_tools_in_docker():
            script = (
                "from pathlib import Path\n"
                "import json, sys\n"
                "path = Path(sys.argv[1])\n"
                "payload = json.loads(sys.stdin.read())\n"
                "content = path.read_text(encoding='utf-8')\n"
                "old_text = payload['old_text']\n"
                "new_text = payload['new_text']\n"
                "if old_text not in content:\n"
                "    raise ValueError(f'Text not found in file: {old_text[:50]}...')\n"
                "path.write_text(content.replace(old_text, new_text, 1), encoding='utf-8')\n"
                "print(f'Successfully edited {sys.argv[1]}', end='')\n"
            )
            return self._run_python_in_docker(
                script,
                [self._workspace_relative_path(file_path)],
                input_text=json.dumps(
                    {"old_text": old_text, "new_text": new_text},
                    ensure_ascii=False,
                ),
            )
        content = self.read_file(file_path)
        if old_text not in content:
            raise ValueError(f"Text not found in file: {old_text[:50]}...")
        new_content = content.replace(old_text, new_text, 1)
        self.write_file(file_path, new_content)
        return f"Successfully edited {file_path}"

    def list_directory(
        self,
        dir_path: str = "",
        file_path: Optional[str] = None,
    ) -> List[str]:
        if file_path and not dir_path:
            dir_path = file_path
        if self._run_workspace_tools_in_docker():
            script = (
                "from pathlib import Path\n"
                "import json, sys\n"
                "root = Path(sys.argv[1])\n"
                "if not root.exists():\n"
                "    raise FileNotFoundError(f'Directory not found: {sys.argv[1]}')\n"
                "items = []\n"
                "for item in sorted(root.iterdir()):\n"
                "    rel_path = item.as_posix()\n"
                "    if item.is_dir():\n"
                "        items.append(f'{rel_path}/')\n"
                "    else:\n"
                "        items.append(f'{rel_path} ({item.stat().st_size} bytes)')\n"
                "print(json.dumps(items, ensure_ascii=False), end='')\n"
            )
            output = self._run_python_in_docker(script, [self._workspace_relative_path(dir_path)])
            return json.loads(output or "[]")
        full_path = self._resolve_workspace_path(dir_path)
        if not full_path.exists():
            raise FileNotFoundError(f"Directory not found: {dir_path}")
        self._ensure_in_workspace(full_path)
        items = []
        for item in sorted(full_path.iterdir()):
            rel_path = item.relative_to(self.workspace_dir).as_posix()
            if item.is_dir():
                items.append(f"{rel_path}/")
            else:
                items.append(f"{rel_path} ({item.stat().st_size} bytes)")
        return items

    def search_code(self, pattern: str, file_pattern: Optional[str] = None) -> List[Dict[str, Any]]:
        if self._run_workspace_tools_in_docker():
            script = (
                "from pathlib import Path; import fnmatch, json, sys; "
                "needle = sys.argv[1].lower(); "
                "file_pattern = None if sys.argv[2] == '__NONE__' else sys.argv[2]; "
                "results = []; "
                "\nfor file_path in Path('.').rglob('*'):\n"
                "    if not file_path.is_file():\n"
                "        continue\n"
                "    rel_path = file_path.relative_to(Path('.')).as_posix()\n"
                "    if file_pattern and not (fnmatch.fnmatch(file_path.name, file_pattern) or fnmatch.fnmatch(rel_path, file_pattern)):\n"
                "        continue\n"
                "    try:\n"
                "        content = file_path.read_text(encoding='utf-8')\n"
                "    except Exception:\n"
                "        continue\n"
                "    for line_number, line in enumerate(content.splitlines(), 1):\n"
                "        if needle in line.lower():\n"
                "            results.append({'file': rel_path, 'line': line_number, 'content': line.strip()})\n"
                "            if len(results) >= 50:\n"
                "                print(json.dumps(results, ensure_ascii=False), end='')\n"
                "                raise SystemExit(0)\n"
                "print(json.dumps(results, ensure_ascii=False), end='')"
            )
            output = self._run_python_in_docker(
                script,
                [pattern, file_pattern or "__NONE__"],
            )
            return json.loads(output or "[]")
        results: List[Dict[str, Any]] = []
        if file_pattern:
            files = list(self.workspace_dir.glob(f"**/{file_pattern}"))
        else:
            files = [candidate for candidate in self.workspace_dir.rglob("*") if candidate.is_file()]
        for file_path in files:
            try:
                self._ensure_in_workspace(file_path)
                content = file_path.read_text(encoding="utf-8")
            except Exception:
                continue
            for line_number, line in enumerate(content.splitlines(), 1):
                if pattern.lower() in line.lower():
                    results.append(
                        {
                            "file": str(file_path.relative_to(self.workspace_dir)),
                            "line": line_number,
                            "content": line.strip(),
                        }
                    )
                    if len(results) >= 50:
                        return results
        return results

    def run_command(self, command: str, timeout: Optional[float] = None) -> str:
        timeout_seconds = self._command_timeout_seconds()
        if timeout is not None:
            try:
                timeout_seconds = float(timeout)
            except (TypeError, ValueError):
                timeout_seconds = self._command_timeout_seconds()
        try:
            if self._command_runner_mode() == "docker":
                self._ensure_docker_container()
                self._maybe_run_docker_bootstrap("before_first_command")
                return self._docker_exec_command(command, timeout=timeout_seconds)
            result = subprocess.run(
                command,
                shell=True,
                cwd=self.workspace_dir,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"Command timed out after {int(timeout_seconds)} seconds"
            )
        except Exception as exc:
            raise RuntimeError(f"Error executing command: {str(exc)}") from exc
        output = result.stdout
        if result.stderr:
            output += f"\nSTDERR:\n{result.stderr}"
        if result.returncode != 0:
            raise RuntimeError(
                output or f"Command failed with exit code {result.returncode}"
            )
        return output or "Command executed successfully (no output)"

    def close(self) -> None:
        self._stop_docker_container()
        for client in self._mcp_clients.values():
            client.close()

    def web_search(self, query: str) -> str:
        searxng_url = self.config.get("api", {}).get("searxng_url")
        if not searxng_url:
            searxng_url = os.getenv("SEARXNG_URL", "http://localhost:your-searxng-port")
        response = requests.get(
            f"{searxng_url}/search",
            params={"q": query, "format": "json"},
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
        results: List[str] = []
        for result in data.get("results", [])[:5]:
            results.append(
                f"Title: {result.get('title')}\nURL: {result.get('url')}\nSnippet: {result.get('content')}\n"
            )
        if not results:
            return "No results found."
        return "\n".join(results)

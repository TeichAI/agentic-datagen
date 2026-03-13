from __future__ import annotations

import importlib
import inspect
import json
import os
import subprocess
import threading
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


class ToolRegistry:
    def __init__(self, workspace_dir: Path, config: Optional[Dict[str, Any]] = None):
        self.workspace_dir = workspace_dir
        self.config = config or {}
        self._tools: Dict[str, RegisteredTool] = {}
        self._mcp_clients: Dict[str, MCPHTTPToolClient] = {}
        self._register_builtin_tools()
        self._register_custom_python_tools()
        self._register_mcp_tools()

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

    def _register_builtin_tools(self) -> None:
        builtin_specs = [
            {
                "name": "read_file",
                "description": "Read a UTF-8 text file from the workspace before editing or to inspect generated code.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the file relative to the workspace root.",
                        }
                    },
                    "required": ["file_path"],
                },
                "handler": self.read_file,
            },
            {
                "name": "write_file",
                "description": "Create or overwrite a workspace file with the provided UTF-8 content. Parent directories are created automatically.",
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
                "description": "Replace the first occurrence of old_text with new_text in an existing workspace file.",
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
                "description": "List files and directories in the workspace or a relative subdirectory to understand project structure.",
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
                "description": "Case-insensitive substring search across workspace files, returning file, line number, and matched content.",
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
                "description": "Execute a shell command inside the workspace and return stdout plus stderr. Useful for builds, tests, and linters.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "Shell command to execute inside the workspace directory.",
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

    def read_file(self, file_path: str) -> str:
        full_path = self.workspace_dir / file_path
        if not full_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        self._ensure_in_workspace(full_path)
        return full_path.read_text(encoding="utf-8")

    def write_file(self, file_path: str, content: str) -> str:
        full_path = self.workspace_dir / file_path
        self._ensure_in_workspace(full_path)
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content, encoding="utf-8")
        return f"Successfully wrote {len(content)} characters to {file_path}"

    def edit_file(self, file_path: str, old_text: str, new_text: str) -> str:
        content = self.read_file(file_path)
        if old_text not in content:
            raise ValueError(f"Text not found in file: {old_text[:50]}...")
        new_content = content.replace(old_text, new_text, 1)
        self.write_file(file_path, new_content)
        return f"Successfully edited {file_path}"

    def list_directory(self, dir_path: str = "") -> List[str]:
        full_path = self.workspace_dir / dir_path if dir_path else self.workspace_dir
        if not full_path.exists():
            raise FileNotFoundError(f"Directory not found: {dir_path}")
        self._ensure_in_workspace(full_path)
        items = []
        for item in sorted(full_path.iterdir()):
            rel_path = item.relative_to(self.workspace_dir)
            if item.is_dir():
                items.append(f"{rel_path}/")
            else:
                items.append(f"{rel_path} ({item.stat().st_size} bytes)")
        return items

    def search_code(self, pattern: str, file_pattern: Optional[str] = None) -> List[Dict[str, Any]]:
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

    def run_command(self, command: str) -> str:
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=self.workspace_dir,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except subprocess.TimeoutExpired:
            return "Error: Command timed out after 30 seconds"
        except Exception as exc:
            return f"Error executing command: {str(exc)}"
        output = result.stdout
        if result.stderr:
            output += f"\nSTDERR:\n{result.stderr}"
        return output or "Command executed successfully (no output)"

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

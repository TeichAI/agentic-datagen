import json
import os
import subprocess
import requests
from pathlib import Path
from typing import Any, Dict, List, Optional


class ToolRegistry:
    """Registry of available tools for the agentic system."""

    def __init__(self, workspace_dir: Path, config: Optional[Dict[str, Any]] = None):
        self.workspace_dir = workspace_dir
        self.config = config or {}
        self.tools = {
            "read_file": self.read_file,
            "write_file": self.write_file,
            "edit_file": self.edit_file,
            "list_directory": self.list_directory,
            "search_code": self.search_code,
            "run_command": self.run_command,
            "web_search": self.web_search,
        }

    def get_tool_definitions(self, enabled_tools: List[str]) -> List[Dict[str, Any]]:
        """Get OpenAI-compatible tool definitions for enabled tools."""
        definitions = []

        if "read_file" in enabled_tools:
            definitions.append(
                {
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "description": "Read the contents of a file in the workspace",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "file_path": {
                                    "type": "string",
                                    "description": "Path to the file relative to workspace root",
                                }
                            },
                            "required": ["file_path"],
                        },
                    },
                }
            )

        if "write_file" in enabled_tools:
            definitions.append(
                {
                    "type": "function",
                    "function": {
                        "name": "write_file",
                        "description": "Write content to a file in the workspace",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "file_path": {
                                    "type": "string",
                                    "description": "Path to the file relative to workspace root",
                                },
                                "content": {
                                    "type": "string",
                                    "description": "Content to write to the file",
                                },
                            },
                            "required": ["file_path", "content"],
                        },
                    },
                }
            )

        if "edit_file" in enabled_tools:
            definitions.append(
                {
                    "type": "function",
                    "function": {
                        "name": "edit_file",
                        "description": "Edit a file by replacing old_text with new_text",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "file_path": {
                                    "type": "string",
                                    "description": "Path to the file relative to workspace root",
                                },
                                "old_text": {
                                    "type": "string",
                                    "description": "Text to replace",
                                },
                                "new_text": {
                                    "type": "string",
                                    "description": "New text to insert",
                                },
                            },
                            "required": ["file_path", "old_text", "new_text"],
                        },
                    },
                }
            )

        if "list_directory" in enabled_tools:
            definitions.append(
                {
                    "type": "function",
                    "function": {
                        "name": "list_directory",
                        "description": "List files and directories in a path",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "dir_path": {
                                    "type": "string",
                                    "description": "Directory path relative to workspace root (empty for root)",
                                }
                            },
                            "required": [],
                        },
                    },
                }
            )

        if "search_code" in enabled_tools:
            definitions.append(
                {
                    "type": "function",
                    "function": {
                        "name": "search_code",
                        "description": "Search for text patterns in workspace files",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "pattern": {
                                    "type": "string",
                                    "description": "Text pattern to search for",
                                },
                                "file_pattern": {
                                    "type": "string",
                                    "description": "Optional file pattern (e.g., '*.py')",
                                },
                            },
                            "required": ["pattern"],
                        },
                    },
                }
            )

        if "run_command" in enabled_tools:
            definitions.append(
                {
                    "type": "function",
                    "function": {
                        "name": "run_command",
                        "description": "Execute a shell command in the workspace",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "command": {
                                    "type": "string",
                                    "description": "Command to execute",
                                }
                            },
                            "required": ["command"],
                        },
                    },
                }
            )

        if "web_search" in enabled_tools:
            definitions.append(
                {
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "description": "Search the web for information",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "Search query",
                                }
                            },
                            "required": ["query"],
                        },
                    },
                }
            )

        return definitions

    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool and return the result."""
        if tool_name not in self.tools:
            return {"error": f"Unknown tool: {tool_name}"}

        try:
            result = self.tools[tool_name](**arguments)
            return {"success": True, "result": result}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def read_file(self, file_path: str) -> str:
        """Read file contents."""
        full_path = self.workspace_dir / file_path
        if not full_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if not str(full_path.resolve()).startswith(str(self.workspace_dir.resolve())):
            raise PermissionError("Access denied: path outside workspace")

        return full_path.read_text(encoding="utf-8")

    def write_file(self, file_path: str, content: str) -> str:
        """Write content to file."""
        full_path = self.workspace_dir / file_path

        if not str(full_path.resolve()).startswith(str(self.workspace_dir.resolve())):
            raise PermissionError("Access denied: path outside workspace")

        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content, encoding="utf-8")
        return f"Successfully wrote {len(content)} characters to {file_path}"

    def edit_file(self, file_path: str, old_text: str, new_text: str) -> str:
        """Edit file by replacing text."""
        content = self.read_file(file_path)

        if old_text not in content:
            raise ValueError(f"Text not found in file: {old_text[:50]}...")

        new_content = content.replace(old_text, new_text, 1)
        self.write_file(file_path, new_content)
        return f"Successfully edited {file_path}"

    def list_directory(self, dir_path: str = "") -> List[str]:
        """List directory contents."""
        full_path = self.workspace_dir / dir_path if dir_path else self.workspace_dir

        if not full_path.exists():
            raise FileNotFoundError(f"Directory not found: {dir_path}")

        if not str(full_path.resolve()).startswith(str(self.workspace_dir.resolve())):
            raise PermissionError("Access denied: path outside workspace")

        items = []
        for item in sorted(full_path.iterdir()):
            rel_path = item.relative_to(self.workspace_dir)
            if item.is_dir():
                items.append(f"{rel_path}/")
            else:
                size = item.stat().st_size
                items.append(f"{rel_path} ({size} bytes)")

        return items

    def search_code(
        self, pattern: str, file_pattern: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Search for pattern in files."""
        results = []

        if file_pattern:
            files = list(self.workspace_dir.glob(f"**/{file_pattern}"))
        else:
            files = [f for f in self.workspace_dir.rglob("*") if f.is_file()]

        for file_path in files:
            try:
                content = file_path.read_text(encoding="utf-8")
                lines = content.split("\n")

                for line_num, line in enumerate(lines, 1):
                    if pattern.lower() in line.lower():
                        results.append(
                            {
                                "file": str(file_path.relative_to(self.workspace_dir)),
                                "line": line_num,
                                "content": line.strip(),
                            }
                        )
            except Exception:
                continue

        return results[:50]

    def run_command(self, command: str) -> str:
        """Execute shell command."""
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=self.workspace_dir,
                capture_output=True,
                text=True,
                timeout=30,
            )

            output = result.stdout
            if result.stderr:
                output += f"\nSTDERR:\n{result.stderr}"

            return output or "Command executed successfully (no output)"
        except subprocess.TimeoutExpired:
            return "Error: Command timed out after 30 seconds"
        except Exception as e:
            return f"Error executing command: {str(e)}"

    def web_search(self, query: str) -> str:
        """Search the web using SearXNG."""
        # Use SEARXNG_URL from config if available, otherwise environment or default
        searxng_url = self.config.get("api", {}).get("searxng_url")
        if not searxng_url:
            searxng_url = os.getenv("SEARXNG_URL", "http://localhost:your-searxng-port")

        try:
            response = requests.get(
                f"{searxng_url}/search",
                params={
                    "q": query,
                    "format": "json",
                },
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()

            results = []
            for result in data.get("results", [])[:5]:
                results.append(
                    f"Title: {result.get('title')}\nURL: {result.get('url')}\nSnippet: {result.get('content')}\n"
                )

            if not results:
                return "No results found."

            return "\n".join(results)
        except Exception as e:
            return f"Error performing web search: {str(e)}"

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


def workspace_snapshot(subdir: str = "", limit: int = 20, context: Dict[str, Any] | None = None) -> Dict[str, Any]:
    context = context or {}
    workspace_dir = Path(context.get("workspace_dir") or ".")
    target = workspace_dir / subdir if subdir else workspace_dir
    target = target.resolve()
    workspace_dir = workspace_dir.resolve()
    if not str(target).startswith(str(workspace_dir)):
        raise PermissionError("Access denied: path outside workspace")
    if not target.exists() or not target.is_dir():
        raise FileNotFoundError(f"Directory not found: {subdir}")
    items = []
    for item in sorted(target.iterdir())[: max(1, limit)]:
        relative = item.relative_to(workspace_dir)
        items.append(
            {
                "path": str(relative),
                "type": "directory" if item.is_dir() else "file",
                "size": item.stat().st_size if item.is_file() else None,
            }
        )
    return {
        "workspace": str(workspace_dir),
        "target": str(target.relative_to(workspace_dir)) if target != workspace_dir else ".",
        "items": items,
    }


TOOLS = [
    {
        "name": "workspace_snapshot",
        "description": "Return a compact structured snapshot of files and directories in the workspace or a subdirectory.",
        "parameters": {
            "type": "object",
            "properties": {
                "subdir": {
                    "type": "string",
                    "description": "Optional relative directory path inside the workspace.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of items to include in the snapshot.",
                    "minimum": 1,
                    "maximum": 200,
                    "default": 20,
                },
            },
            "required": [],
        },
        "handler": workspace_snapshot,
    }
]

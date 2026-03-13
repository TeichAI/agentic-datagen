from __future__ import annotations

import hashlib
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


class RunManifest:
    def __init__(self, path: Path, run_id: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self._lock = threading.Lock()
        self._data = self._load()
        if not self._data.get("run_id"):
            self._data["run_id"] = run_id
        self._data.setdefault("entries", {})
        self._data.setdefault("summary", {})
        self._data.setdefault("created_at", self._now())
        self._data["updated_at"] = self._now()
        self._recompute_summary()
        self._persist_locked()

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _load(self) -> Dict[str, Any]:
        if not self.path.exists():
            return {}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def _persist_locked(self) -> None:
        self._data["updated_at"] = self._now()
        tmp_path = self.path.with_suffix(f"{self.path.suffix}.tmp")
        tmp_path.write_text(
            json.dumps(self._data, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        tmp_path.replace(self.path)

    def _recompute_summary(self) -> None:
        entries = self._data.get("entries", {}).values()
        status_counts: Dict[str, int] = {}
        for entry in entries:
            status = str(entry.get("status") or "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1
        self._data["summary"] = {
            "total_prompts": len(self._data.get("entries", {})),
            "status_counts": status_counts,
        }

    def make_prompt_hash(self, prompt: str) -> str:
        return hashlib.sha256(prompt.strip().encode("utf-8")).hexdigest()

    def make_prompt_id(self, index: int, prompt: str) -> str:
        return f"prompt_{index:06d}_{self.make_prompt_hash(prompt)[:12]}"

    def _entry_preview(self, prompt: str, limit: int = 120) -> str:
        normalized = " ".join(prompt.strip().split())
        return normalized[:limit]

    def seed_prompts(self, prompts: List[str], completed_prompts: Set[str]) -> None:
        with self._lock:
            entries = self._data.setdefault("entries", {})
            for index, prompt in enumerate(prompts):
                prompt_id = self.make_prompt_id(index, prompt)
                entry = entries.get(prompt_id, {})
                prompt_hash = self.make_prompt_hash(prompt)
                entry.setdefault("prompt_id", prompt_id)
                entry.setdefault("index", index)
                entry.setdefault("prompt_hash", prompt_hash)
                entry.setdefault("prompt_preview", self._entry_preview(prompt))
                entry.setdefault("attempt_count", 0)
                entry.setdefault("workspace_dir", None)
                if prompt.strip() in completed_prompts:
                    entry["status"] = "completed"
                    entry["completed"] = True
                    entry.setdefault("dataset_route", "dataset")
                else:
                    entry.setdefault("status", "pending")
                    entry.setdefault("completed", False)
                entries[prompt_id] = entry
            self._recompute_summary()
            self._persist_locked()

    def mark_running(
        self,
        prompt_id: str,
        index: int,
        prompt: str,
        workspace_dir: Path,
    ) -> None:
        with self._lock:
            entries = self._data.setdefault("entries", {})
            entry = entries.get(prompt_id, {})
            entry["prompt_id"] = prompt_id
            entry["index"] = index
            entry["prompt_hash"] = self.make_prompt_hash(prompt)
            entry["prompt_preview"] = self._entry_preview(prompt)
            entry["status"] = "running"
            entry["workspace_dir"] = str(workspace_dir)
            entry["attempt_count"] = int(entry.get("attempt_count") or 0) + 1
            entry["started_at"] = self._now()
            entry["last_error"] = None
            entries[prompt_id] = entry
            self._recompute_summary()
            self._persist_locked()

    def mark_result(
        self,
        prompt_id: str,
        *,
        status: str,
        completed: bool,
        retryable: bool,
        error: Optional[str],
        turns: Optional[int],
        tool_calls_count: Optional[int],
        usage: Optional[Dict[str, Any]],
        workspace_dir: Optional[Path],
    ) -> None:
        with self._lock:
            entries = self._data.setdefault("entries", {})
            entry = entries.get(prompt_id, {})
            entry["status"] = status
            entry["completed"] = completed
            entry["retryable"] = retryable
            entry["last_error"] = error
            entry["turns"] = turns
            entry["tool_calls_count"] = tool_calls_count
            entry["usage"] = usage or {}
            if workspace_dir is not None:
                entry["workspace_dir"] = str(workspace_dir)
            entry["finished_at"] = self._now()
            entries[prompt_id] = entry
            self._recompute_summary()
            self._persist_locked()

    def set_route(self, prompt_id: Optional[str], route: str) -> None:
        if not prompt_id:
            return
        with self._lock:
            entries = self._data.setdefault("entries", {})
            entry = entries.get(prompt_id)
            if not entry:
                return
            entry["dataset_route"] = route
            entries[prompt_id] = entry
            self._recompute_summary()
            self._persist_locked()

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return json.loads(json.dumps(self._data))

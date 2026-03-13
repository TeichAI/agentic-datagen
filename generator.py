import json
import logging
import os
import shutil
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import yaml

from agent_session import AgentSession
from formatter import Formatter
from run_manifest import RunManifest


class AgenticDatasetGenerator:
    """Main orchestrator for agentic dataset generation."""

    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        self.formatter = Formatter()
        self.write_lock = threading.Lock()
        self.run_id = self.config.get("output", {}).get("run_id") or datetime.now(
            timezone.utc
        ).strftime("run_%Y%m%dT%H%M%SZ")

        self.api_key = self._get_api_key()
        self.config["api"]["api_key"] = self.api_key

        self.base_workspace_dir = Path(self.config["workspace"]["base_dir"])
        self.base_workspace_dir.mkdir(parents=True, exist_ok=True)

        self.output_file = Path(self.config["output"]["dataset_file"])
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        manifest_path = self.config.get("output", {}).get("run_manifest_file")
        if manifest_path:
            self.run_manifest_file = Path(manifest_path)
        else:
            self.run_manifest_file = self.output_file.with_suffix(".manifest.json")
        self.run_manifest_file.parent.mkdir(parents=True, exist_ok=True)
        self.run_manifest = RunManifest(self.run_manifest_file, run_id=self.run_id)

        # Handle overwrite mode initialization
        if not self.config.get("output", {}).get("append_mode", True):
            if self.output_file.exists():
                self.output_file.unlink()

        error_output_path = self.config.get("output", {}).get("error_dataset_file")
        self.error_output_file = None
        if error_output_path:
            self.error_output_file = Path(error_output_path)
            self.error_output_file.parent.mkdir(parents=True, exist_ok=True)

        if self.config.get("output", {}).get("sanitize_existing_dataset", True):
            self._sanitize_output_dataset()

        # Initialize global tool definitions if available
        self.enabled_tools = self.config["agent"].get("tools_enabled", [])
        from tools import ToolRegistry

        temp_registry = ToolRegistry(Path("."), self.config)
        self.tool_definitions = temp_registry.get_tool_definitions(self.enabled_tools)

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        log_config = self.config.get("logging", {})
        level = getattr(logging, log_config.get("level", "INFO"))

        logger = logging.getLogger("agentic_datagen")
        logger.setLevel(level)

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        if log_config.get("console", True):
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        if "log_file" in log_config:
            file_handler = logging.FileHandler(log_config["log_file"])
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        return logger

    def _get_api_key(self) -> str:
        """Get API key from config or environment."""
        api_config = self.config["api"]

        # 1. Direct key in config
        if "api_key" in api_config and api_config["api_key"]:
            api_key = api_config["api_key"]
            self.logger.info(
                f"Using API Key from config: {api_key[:4]}...{api_key[-4:]}"
            )
            return api_key

        # 2. Environment variable
        env_var = api_config.get("api_key_env", "OPENROUTER_API_KEY")
        api_key = os.getenv(env_var)

        if not api_key:
            from dotenv import dotenv_values

            api_key = dotenv_values(".env").get(env_var)

        if not api_key:
            old_env = Path("old/.env")
            if old_env.exists():
                from dotenv import dotenv_values

                api_key = dotenv_values(old_env).get(env_var)

        if not api_key:
            raise ValueError(
                f"Missing API key. Provide 'api_key' in config or set {env_var} environment variable."
            )

        self.logger.info(
            f"Using API Key from {env_var}: {api_key[:4]}...{api_key[-4:]}"
        )
        return api_key

    def _load_prompts(self) -> List[str]:
        """Load prompts from configured source."""
        from utils import load_prompts

        prompts_config = self.config["prompts"]
        source_path = Path(prompts_config["source"])

        if self.error_output_file and source_path.resolve() == self.error_output_file.resolve():
            raise ValueError(
                "Prompt source and error_dataset_file must be different files"
            )

        prompts = load_prompts(source_path)

        if prompts_config.get("shuffle", False):
            import random

            random.shuffle(prompts)

        limit = prompts_config.get("limit")
        if limit and limit > 0:
            prompts = prompts[:limit]

        return prompts

    def _load_completed_prompts(self) -> Set[str]:
        """Load prompts that have already been processed."""
        completed = set()

        if not self.output_file.exists():
            return completed

        with self.output_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    entry = json.loads(line)
                    if not self.formatter.validate_entry(entry, require_completion=True):
                        continue
                    prompt = entry.get("prompt")
                    if prompt:
                        completed.add(prompt.strip())
                except json.JSONDecodeError:
                    continue

        return completed

    def _sanitize_output_dataset(self) -> None:
        if not self.output_file.exists():
            return

        temp_file = self.output_file.with_name(f"{self.output_file.name}.tmp")
        valid_entries = 0
        removed_entries = 0

        with self.output_file.open("r", encoding="utf-8") as src, temp_file.open(
            "w", encoding="utf-8"
        ) as dst:
            for line in src:
                stripped = line.strip()
                if not stripped:
                    continue

                try:
                    entry = json.loads(stripped)
                except json.JSONDecodeError:
                    removed_entries += 1
                    continue

                if not self.formatter.validate_entry(entry, require_completion=True):
                    removed_entries += 1
                    continue

                dst.write(json.dumps(entry, ensure_ascii=False) + "\n")
                valid_entries += 1

        if removed_entries:
            temp_file.replace(self.output_file)
            self.logger.warning(
                "Removed %s contaminated entries from %s; retained %s clean completed entries",
                removed_entries,
                self.output_file,
                valid_entries,
            )
            return

        if temp_file.exists():
            temp_file.unlink()

    def _create_workspace(self, session_id: str) -> Path:
        """Create a workspace directory for a session."""
        workspace_dir = self.base_workspace_dir / session_id
        workspace_dir.mkdir(parents=True, exist_ok=True)
        return workspace_dir

    def _cleanup_workspace(self, workspace_dir: Path):
        """Remove workspace directory."""
        if workspace_dir.exists():
            shutil.rmtree(workspace_dir)

    def _process_prompt(self, prompt: str, index: int) -> Optional[Dict[str, Any]]:
        """Process a single prompt and return formatted entry."""
        prompt_id = self.run_manifest.make_prompt_id(index, prompt)
        session_id = f"session_{index:06d}"
        workspace_dir = self._create_workspace(session_id)
        session: Optional[AgentSession] = None

        self.logger.info(f"Processing prompt {index}: {prompt[:80]}...")
        self.run_manifest.mark_running(prompt_id, index, prompt, workspace_dir)

        try:
            session = AgentSession(
                prompt=prompt,
                workspace_dir=workspace_dir,
                api_config=self.config["api"],
                agent_config=self.config["agent"],
                session_id=session_id,
                prompt_id=prompt_id,
                run_id=self.run_id,
                runtime_config=self.config,
            )

            session_data = session.run()

            is_error = bool(session_data.get("error"))
            is_completed = bool(session_data.get("completed") and not session_data.get("error"))
            status = "completed"
            if not is_completed and session_data.get("retryable"):
                status = "retryable_error"
            elif is_error:
                status = "fatal_error"
            elif not is_completed:
                status = "incomplete"
            self.run_manifest.mark_result(
                prompt_id,
                status=status,
                completed=is_completed,
                retryable=bool(session_data.get("retryable")),
                error=session_data.get("error"),
                turns=session_data.get("turns"),
                tool_calls_count=len(session_data.get("tool_calls") or []),
                usage=session_data.get("usage"),
                workspace_dir=workspace_dir,
            )
            if is_error:
                self.logger.error(f"Session error: {session_data['error']}")
                if self.config["workspace"].get("preserve_on_error", True):
                    self.logger.info(f"Preserving workspace: {workspace_dir}")
                else:
                    self._cleanup_workspace(workspace_dir)

            formatted_entry = self.formatter.format_session(session_data)

            # Add tools column
            formatted_entry["tools"] = self.tool_definitions

            # Reorder columns for output readability
            formatted_entry = {
                "prompt": formatted_entry.get("prompt"),
                "tools": formatted_entry.get("tools"),
                "messages": formatted_entry.get("messages"),
                "metadata": formatted_entry.get("metadata"),
                "usage": formatted_entry.get("usage"),
            }

            if not self.formatter.validate_entry(
                formatted_entry, require_completion=is_completed
            ):
                self.logger.error("Entry validation failed")
                return None

            if self.config["workspace"].get("cleanup", True) and is_completed:
                self._cleanup_workspace(workspace_dir)
            else:
                self.logger.info(f"Preserving workspace: {workspace_dir}")

            return formatted_entry

        except Exception as e:
            self.logger.error(f"Error processing prompt: {e}", exc_info=True)
            self.run_manifest.mark_result(
                prompt_id,
                status="fatal_error",
                completed=False,
                retryable=False,
                error=str(e),
                turns=None,
                tool_calls_count=None,
                usage=None,
                workspace_dir=workspace_dir,
            )
            if self.config["workspace"].get("preserve_on_error", True):
                self.logger.info(f"Preserving workspace: {workspace_dir}")
            else:
                self._cleanup_workspace(workspace_dir)
            return None
        finally:
            if session is not None:
                session.close()

    def _append_to_dataset(self, entry: Dict[str, Any]):
        """Append entry to dataset file."""
        jsonl_line = self.formatter.to_jsonl_line(entry)

        # Always append, because we handled truncation in __init__
        with self.write_lock:
            with self.output_file.open("a", encoding="utf-8") as f:
                f.write(jsonl_line + "\n")

    def _append_to_error_dataset(self, entry: Dict[str, Any]):
        """Append entry to error dataset file."""
        if not self.error_output_file:
            return

        jsonl_line = self.formatter.to_jsonl_line(entry)
        with self.write_lock:
            with self.error_output_file.open("a", encoding="utf-8") as f:
                f.write(jsonl_line + "\n")

    def _is_training_safe_entry(self, entry: Dict[str, Any]) -> bool:
        return self.formatter.validate_entry(entry, require_completion=True)

    def _route_entry(self, entry: Dict[str, Any]) -> None:
        prompt_id = (entry.get("metadata") or {}).get("prompt_id")
        if self._is_training_safe_entry(entry):
            self._append_to_dataset(entry)
            self.run_manifest.set_route(prompt_id, "dataset")
            return

        if self.error_output_file and self.formatter.validate_entry(entry):
            self._append_to_error_dataset(entry)
            self.run_manifest.set_route(prompt_id, "error_dataset")

    def generate(self):
        """Main generation loop."""
        from tqdm import tqdm

        self.logger.info("Starting agentic dataset generation")

        prompts = self._load_prompts()
        self.logger.info(f"Loaded {len(prompts)} prompts")

        if self.config["processing"].get("resume", True):
            completed = self._load_completed_prompts()
            self.logger.info(f"Found {len(completed)} completed prompts")
            self.run_manifest.seed_prompts(prompts, completed)

            prompts_to_process = [
                (i, p) for i, p in enumerate(prompts) if p.strip() not in completed
            ]
        else:
            self.run_manifest.seed_prompts(prompts, set())
            prompts_to_process = list(enumerate(prompts))

        if not prompts_to_process:
            self.logger.info("No prompts to process")
            return

        self.logger.info(f"Processing {len(prompts_to_process)} prompts")

        concurrency = self.config["processing"].get("concurrency", 1)
        total_prompts = len(prompts_to_process)

        # Tracking metrics
        self.total_cost = 0.0
        self.total_tokens = 0

        pbar = tqdm(total=total_prompts, desc="Generating Dataset")

        def update_pbar(entry):
            if entry and "usage" in entry:
                self.total_cost += entry["usage"].get("cost", 0.0)
                self.total_tokens += entry["usage"].get("total_tokens", 0)

            pbar.set_postfix(
                {"cost": f"${self.total_cost:.4f}", "tokens": f"{self.total_tokens:,}"}
            )
            pbar.update(1)

        if concurrency <= 1:
            for index, prompt in prompts_to_process:
                entry = self._process_prompt(prompt, index)

                if entry:
                    self._route_entry(entry)
                    update_pbar(entry)
                else:
                    pbar.update(1)
        else:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = {
                    executor.submit(self._process_prompt, prompt, index): (
                        index,
                        prompt,
                    )
                    for index, prompt in prompts_to_process
                }

                for future in as_completed(futures):
                    try:
                        entry = future.result()
                        if entry:
                            self._route_entry(entry)
                            update_pbar(entry)
                    except Exception as e:
                        self.logger.error(f"Error in future: {e}")
                        pbar.update(1)

        pbar.close()
        self.logger.info("Dataset generation complete")
        self.logger.info(f"Total Cost: ${self.total_cost:.4f}")
        self.logger.info(f"Total Tokens: {self.total_tokens:,}")
        self.logger.info(f"Output saved to: {self.output_file}")


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate agentic datasets with tool-calling capabilities"
    )
    parser.add_argument(
        "-c", "--config", required=True, help="Path to configuration YAML file"
    )

    args = parser.parse_args()

    try:
        generator = AgenticDatasetGenerator(args.config)
        generator.generate()
    except Exception as e:
        import traceback

        traceback.print_exc()
        print(f"Fatal error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

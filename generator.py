import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import yaml

from agent_session import AgentSession
from formatter import Formatter


class AgenticDatasetGenerator:
    """Main orchestrator for agentic dataset generation."""

    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        self.formatter = Formatter()

        self.api_key = self._get_api_key()
        self.config["api"]["api_key"] = self.api_key

        self.base_workspace_dir = Path(self.config["workspace"]["base_dir"])
        self.base_workspace_dir.mkdir(parents=True, exist_ok=True)

        self.output_file = Path(self.config["output"]["dataset_file"])
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

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
                    metadata = entry.get("metadata", {})
                    prompt = metadata.get("prompt")
                    if prompt:
                        completed.add(prompt.strip())
                except json.JSONDecodeError:
                    continue

        return completed

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
        session_id = f"session_{index:06d}"
        workspace_dir = self._create_workspace(session_id)

        self.logger.info(f"Processing prompt {index}: {prompt[:80]}...")

        try:
            session = AgentSession(
                prompt=prompt,
                workspace_dir=workspace_dir,
                api_config=self.config["api"],
                agent_config=self.config["agent"],
                session_id=session_id,
            )

            session_data = session.run()
            session.close()

            if "error" in session_data:
                self.logger.error(f"Session error: {session_data['error']}")
                if self.config["workspace"].get("preserve_on_error", True):
                    self.logger.info(f"Preserving workspace: {workspace_dir}")
                else:
                    self._cleanup_workspace(workspace_dir)
                return None

            formatted_entry = self.formatter.format_session(session_data)

            # Add tools column
            formatted_entry["tools"] = self.tool_definitions

            # Hardcoded validation and format
            if not self.formatter.validate_entry(formatted_entry):
                self.logger.error("Entry validation failed")
                return None

            if self.config["workspace"].get("cleanup", True):
                self._cleanup_workspace(workspace_dir)
            else:
                self.logger.info(f"Preserving workspace: {workspace_dir}")

            return formatted_entry

        except Exception as e:
            self.logger.error(f"Error processing prompt: {e}", exc_info=True)
            if self.config["workspace"].get("preserve_on_error", True):
                self.logger.info(f"Preserving workspace: {workspace_dir}")
            else:
                self._cleanup_workspace(workspace_dir)
            return None

    def _append_to_dataset(self, entry: Dict[str, Any]):
        """Append entry to dataset file."""
        jsonl_line = self.formatter.to_jsonl_line(entry)

        # Default to append mode
        mode = "a" if self.config.get("output", {}).get("append_mode", True) else "w"
        with self.output_file.open(mode, encoding="utf-8") as f:
            f.write(jsonl_line + "\n")

    def generate(self):
        """Main generation loop."""
        from tqdm import tqdm

        self.logger.info("Starting agentic dataset generation")

        prompts = self._load_prompts()
        self.logger.info(f"Loaded {len(prompts)} prompts")

        if self.config["processing"].get("resume", True):
            completed = self._load_completed_prompts()
            self.logger.info(f"Found {len(completed)} completed prompts")

            prompts_to_process = [
                (i, p) for i, p in enumerate(prompts) if p.strip() not in completed
            ]
        else:
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
                    self._append_to_dataset(entry)
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
                            self._append_to_dataset(entry)
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
        print(f"Fatal error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

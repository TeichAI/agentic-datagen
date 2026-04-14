# Agentic Dataset Generator

A tool for creating agentic coding datasets with tool-calling capabilities.

## Overview

This tool generates synthetic agentic datasets by:

1. Loading prompts from a configured source
2. Creating isolated workspaces for each prompt
3. Running an AI agent with tool access
4. Recording all reasoning, tool calls, and responses
5. Validating and appending to a JSONL dataset file

## Features

- **Windsurf/Cursor/Codex-like Tools**: File operations (read, write, edit), directory listing, code search, command execution.
- **Extensible Tool Registry**: Built-in tools, custom Python tools, and MCP-backed HTTP tools can all be enabled from config.
- **Web Search**: Live integration with SearXNG instances.
- **Live Metrics & Progress**: Real-time CLI tracking of cost (USD), token count, and completion status via `tqdm`.
- **Workspace Isolation**: Each prompt gets its own workspace directory (`sandbox/` by default).
- **Session Recording**: Complete multi-turn trajectories including reasoning and tool outputs.
- **Resume Support**: Automatically skips already processed prompts using per-row `prompt_id` tracking.
- **Run Manifest**: Per-prompt manifest with status, attempts, workspace path, usage, and automatic retry state.
- **Duplicate Prompt Preservation**: Repeated prompts in the input source are preserved and processed as distinct rows.
- **Unified CLI**: Generation, QA, and cleanup now live under `cli.py` subcommands.
- **Rich QA & Cleanup Tooling**: Terminal-friendly dataset auditing plus an interactive cleanup CLI for removing bad rows with a backup.
- **Final Dataset Totals**: QA reports post-filter dataset totals including final rows, prompt tokens, completion tokens, total tokens, total cost, and aggregate averages.
- **Optional Docker Session Isolation**: Run either just `run_command` or the full built-in workspace tool surface inside a per-session Docker container to avoid host-environment conflicts and runtime drift.
- **`src/` Package Layout**: Core source now lives under `src/agentic_datagen` with top-level compatibility wrappers.
- **Flexible Prompt Sources**: Accepts `.txt`, `.json`, and `.jsonl` sources.

## Pre-requisites

### OpenRouter API Key

You will need an OpenRouter API key to run the generation. You can get one from [OpenRouter](https://openrouter.ai/keys).

### SearXNG Instance (Optional)

If you want to use the builtin `web_search` tool, you will need a SearXNG instance. You can [set up your own instance](https://docs.searxng.org/admin/installation-docker.html#installation-container) or use a public one.

### Docker (Recommended)

It's recommended to use the Docker session isolation feature to avoid host-environment conflicts and runtime drift. You will need to have Docker installed and running on your system. Docker installation instructions can be found [here](https://docs.docker.com/get-started/get-docker/).

## Installation

```bash
# Clone the repository
git clone https://github.com/TeichAI/agentic_datagen.git
cd agentic_datagen

# Install dependencies
uv pip install -r requirements.txt
```

## Quick Start

```bash
# Create config from example
cp config.example.yaml config.yaml

# Run generation
uv run cli.py generate -c config.yaml
# Audit the final dataset
uv run cli.py qa datasets/Hunter-Alpha-Coding-Agent-SFT.jsonl

# Preview a cleanup plan
uv run cli.py cleanup datasets/Hunter-Alpha-Coding-Agent-SFT.jsonl --max-failed-tool-calls 0 --dry-run
```

## Configuration

The tool uses a simple YAML configuration file. See `config.example.yaml` for the current recommended template.

### Minimal Configuration

```yaml
model:
  provider: "openrouter"
  name: "openrouter/elephant-alpha"
  api_key_env: "OPENROUTER_API_KEY"
  reasoning_effort: "medium"

prompts:
  source: "prompts.txt" # .txt, .jsonl, or .json

tools:
  enabled:
    - read_file
    - write_file
    - run_command
    - web_search
  web_search:
    searxng_url: "http://localhost:your-searxng-port"

output:
  dataset_file: "datasets/agentic_dataset.jsonl"
  run_manifest_file: "datasets/agentic_dataset.manifest.json" # Optional
  append_mode: true
```

### Model Options

```yaml
model:
  provider: "openrouter" # "openrouter" or "openai" are the main OpenAI-compatible options
  base_url: "https://openrouter.ai/api/v1/chat/completions" # Full chat-completions URL, or an OpenAI-compatible root /v1 URL
  api_key_env: "OPENROUTER_API_KEY" # Read API key from env instead of api_key
  name: "openrouter/elephant-alpha" # Model identifier
  reasoning_effort: "medium" # OpenRouter uses reasoning: { effort: ... }; OpenAI-compatible providers use reasoning_effort
  system_prompt: null # Optional system prompt override
  max_turns: 50 # Maximum turns per prompt
  max_retries: 5 # Retries for retryable transport/provider failures
  backoff_base_seconds: 2.0 # Exponential backoff base delay
  backoff_max_seconds: 60.0 # Maximum retry delay
  timeout: 120 # Request timeout in seconds
```

The public config is normalized internally into the generator's runtime config. That keeps the user-facing YAML small while preserving the richer internal defaults needed by the runtime.

Common provider setups:

```yaml
# OpenAI cloud
model:
  provider: "openai"
  name: "gpt-4.1-mini"
  api_key_env: "OPENAI_API_KEY"

# vLLM / llama.cpp / LM Studio / other OpenAI-compatible server
model:
  provider: "openai"
  base_url: "http://localhost:8000/v1"
  name: "Qwen/Qwen2.5-Coder-32B-Instruct"
```

- If you provide `base_url: "http://localhost:8000/v1"`, the runtime automatically expands it to `.../chat/completions`.
- For `provider: "openai"`, the default API key env var becomes `OPENAI_API_KEY`.
- Local OpenAI-compatible endpoints on `localhost` / `127.0.0.1` can be used without an API key.

### Prompt Sources

Supported formats: `.txt`, `.json`, `.jsonl`.

- **Text**: each line is a prompt.
- **JSON/JSONL**: each object can use one of these keys: `prompt`, `input`, `question`, `task`, `query`, or a chat-style `messages` array where `user` messages are extracted as prompts.
- Duplicate prompts are preserved instead of being silently deduplicated.

### Output Files

```yaml
output:
  dataset_file: "datasets/agentic_dataset.jsonl"
  run_manifest_file: "datasets/agentic_dataset.manifest.json" # Optional
  append_mode: true
```

- `dataset_file` stores successful sessions.
- `run_manifest_file` stores one record per prompt with status, attempts, route, and usage metadata.
- Resume behavior is driven by the cleaned `dataset_file` plus the manifest, so retries happen automatically without maintaining a separate error dataset.
- `error_dataset_file` is no longer part of the recommended workflow.
- The main dataset excludes suspiciously shallow completed build/create trajectories, such as very short sessions with only exploratory tool usage and no file-mutating work.

### Tool Configuration

```yaml
tools:
  enabled:
    - read_file
    - write_file
    - edit_file
    - list_directory
    - search_code
    - run_command
    - web_search
    - workspace_snapshot
    - context7:*
  web_search:
    searxng_url: "http://localhost:your-searxng-port"
  custom_python_modules:
    - custom_tools.example_tools
  strict_mcp: false
```

The runtime still supports advanced internals like custom Python tools, MCP servers, and workspace runner overrides, but those do not need to be configured just to get a run working.

By default, workspace file tools and `run_command` execute inside a per-session Docker container for better isolation from host ports, dependencies, and local machine state. Use host mode only as an explicit opt-out when you understand the tradeoffs.

### Advanced: Workspace Command Runner

Built-in workspace tools can execute either on the host or inside a per-session Docker container. Docker with `tool_scope: "all"` is the default path.

```yaml
workspace:
  command_runner:
    mode: "docker"
    tool_scope: "all"
    eager_start: true
    timeout_seconds: 30
    bootstrap_trigger: "before_first_command"
    bootstrap_timeout_seconds: 120
    bootstrap_commands:
      - "if [ -f package.json ]; then npm install; fi"
      - "if [ -f pyproject.toml ]; then uv sync; fi"
    docker_binary: "docker"
    docker_image: "agentic-datagen-session-runtime:latest"
    container_workspace_dir: "/workspace"
    use_host_user: true
    shell: "/bin/sh"
    shell_args: ["-lc"]
```

- Use `tool_scope: "command"` to isolate only `run_command`.
- Use `tool_scope: "all"` to route `read_file`, `write_file`, `edit_file`, `list_directory`, `search_code`, and `run_command` through the session container.
- If you really need host execution, set `workspace.command_runner.mode: "host"` explicitly.
- Use `eager_start: true` if you want each active session to create its container immediately instead of waiting until the first tool call.
- Use `bootstrap_commands` to run one-time per-session setup inside the workspace container.
- Use `bootstrap_trigger: "before_first_command"` when the agent is expected to create project files first and only then install dependencies.
- Use `bootstrap_trigger: "container_start"` when the container should self-prepare immediately on session startup.
- The workspace directory is bind-mounted into the container, so the `sandbox/session_*` files remain visible on the host while tool execution happens inside the container.
- `use_host_user: true` prevents bind-mounted files from becoming `root`-owned on Linux hosts.
- Choose a `docker_image` that matches your workload. The included session runtime image is a good default when you want a fuller Linux dev environment.

 Build the included Ubuntu-based session runtime image with:

```bash
docker build -t agentic-datagen-session-runtime:latest -f src/docker/session-runtime.Dockerfile .
```

The included runtime image preinstalls:

- Node.js 22 and npm
- Python 3 and pip
- Astral `uv` and `uvx`
- git
- build-essential / make / g++
- ripgrep
- sqlite3

### Agent Prompting

If `model.system_prompt` is omitted, the generator uses a short default prompt tuned for code-editing trajectories:

```text
You are a coding agent. Use tools deliberately, inspect before editing, and finish the user's request with working files inside the workspace. When Context7 documentation tools are available and you are working with libraries or frameworks, use Context7 to fetch the latest relevant docs before making library-specific changes.
```

You can override it in config when you want a stricter or domain-specific behavior.

## Usage

```bash
# Generate or resume a run
uv run cli.py generate -c config.yaml

# QA the current dataset
uv run cli.py qa datasets/Hunter-Alpha-Coding-Agent-SFT.jsonl

# Clean the current dataset in place after previewing the plan
uv run cli.py cleanup datasets/Hunter-Alpha-Coding-Agent-SFT.jsonl \
  --max-failed-tool-calls 0 \
  --remove-port-conflicts \
  --yes
```

### Dataset QA validator

Use the validator to audit a generated JSONL dataset for structural issues and quality signals.

```bash
# Rich terminal summary when rich is installed
uv run cli.py qa datasets/Hunter-Alpha-Coding-Agent-SFT.jsonl

# Plain text summary
uv run cli.py qa datasets/Hunter-Alpha-Coding-Agent-SFT.jsonl --plain

# Machine-readable JSON report
uv run cli.py qa datasets/Hunter-Alpha-Coding-Agent-SFT.jsonl --json

# Exit non-zero if any hard errors are found
uv run cli.py qa datasets/Hunter-Alpha-Coding-Agent-SFT.jsonl --fail-on-errors
```

The validator reports:

- entries with `metadata.error`
- entries not ending in a plain `assistant` message
- invalid JSON or invalid message structure
- failed tool call counts per conversation and in aggregate
- suspiciously shallow completed build/create trajectories that are excluded from the main dataset
- final dataset-wide totals such as final rows, prompt tokens, completion tokens, total tokens, total cost, average turns, and average tool calls
- quality signals such as reasoning tags, localhost mentions, and port-conflict mentions

Port conflicts, shallow completions, and failed tools are actionable warnings/errors.

### Interactive dataset cleanup

Use the cleanup script to preview removals, create a backup, and rewrite the dataset.

```bash
# Interactive cleanup with prompts and backup creation
uv run cli.py cleanup datasets/Hunter-Alpha-Coding-Agent-SFT.jsonl

# Non-interactive cleanup for shallow rows and port conflicts
uv run cli.py cleanup datasets/Hunter-Alpha-Coding-Agent-SFT.jsonl \
  --remove-shallow \
  --remove-port-conflicts \
  --yes

# Remove entries with any failed tool calls and low-quality rows, but only preview the plan
uv run cli.py cleanup datasets/Hunter-Alpha-Coding-Agent-SFT.jsonl \
  --max-failed-tool-calls 0 \
  --min-quality-score 95 \
  --dry-run
```

The cleanup script always creates a timestamped backup before rewriting the dataset.

## Available Tools

- **read_file**: Read file contents from workspace
- **write_file**: Write content to a file
- **edit_file**: Replace text in a file
- **list_directory**: List files and directories
- **search_code**: Search for patterns in files
- **run_command**: Execute shell commands (with timeout)
- **web_search**: Search the web using SearXNG

## Custom Tool Quickstart

The runtime now supports three tool sources:

- **Built-in tools** defined by the generator.
- **Custom Python tools** loaded from modules listed in `tools.custom_python_modules`.
- **MCP HTTP tools** discovered from configured MCP servers.

### Add a custom Python tool

1. Create a Python module, for example `src/custom_tools/example_tools.py`.
2. Export either:
   - `TOOLS`: a list of tool spec dictionaries, or
   - `register_tools(registry)`: a function that returns tool specs or registers them directly.
3. Add the module path to `tools.custom_python_modules`.
4. Add the tool name to `tools.enabled`.

Each tool spec must contain:

- `name`
- `description`
- `parameters` (JSON Schema object)
- `handler` (callable)

Example:

```python
from typing import Any, Dict


def workspace_snapshot(limit: int = 20, context: Dict[str, Any] | None = None) -> Dict[str, Any]:
    context = context or {}
    workspace_dir = context["workspace_dir"]
    items = sorted(workspace_dir.iterdir())[:limit]
    return {"items": [item.name for item in items]}


TOOLS = [
    {
        "name": "workspace_snapshot",
        "description": "Return a compact snapshot of files in the workspace.",
        "parameters": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of files to include.",
                    "default": 20,
                }
            },
            "required": [],
        },
        "handler": workspace_snapshot,
    }
]
```

The registry automatically injects these optional handler kwargs when present in the function signature:

- `context`
- `workspace_dir`
- `config`
- `registry`

### Enable MCP HTTP tools

The generator supports MCP tool discovery and invocation over JSON-RPC HTTP transport.

```yaml
tools:
  strict_mcp: false
  mcp_servers:
    context7:
      transport: "http"
      url: "https://mcp.context7.com/mcp"
      timeout: 30
      tool_name_prefix: "context7"
      headers:
        CONTEXT7_API_KEY: "YOUR_API_KEY"
  enabled:
    - context7:*
```

Notes:

- Remote MCP tools are exposed locally as either `mcp__<server>__<tool>` or `<tool_name_prefix>__<tool>`.
- A selector like `context7:*` enables every discovered tool from that MCP server.
- When `tools.strict_mcp` is `false`, unreachable MCP servers are skipped instead of failing the whole run.
- Current support targets MCP JSON-RPC over HTTP.

## Live Metrics & Progress

The tool provides a live CLI progress bar using `tqdm`, tracking:

- **Total Cost**: Real-time USD spend (based on OpenRouter/API usage reporting).
- **Token Count**: Total cumulative input and output tokens.
- **Completion Rate**: Remaining prompts and estimated time to completion.

## Workflow

1. Loading prompts from configured source
2. Creating isolated workspaces for each prompt
3. Running an AI agent with tool access
4. Recording all reasoning, tool calls, and responses
5. Formatting output to match OpenAI structure
6. Validating and appending to a JSONL dataset file
7. Cleaning up workspaces (if configured)

## Error Handling & Retry Workflow

Retry state is now tracked automatically in the manifest instead of a separate error dataset.

- Successful, training-safe rows are appended to `dataset_file`.
- Failed or filtered rows remain absent from `dataset_file` and stay pending/retryable in the manifest.
- On the next resume, the generator automatically retries rows that are not represented in the cleaned dataset.
- This keeps the upload-bound dataset clean without forcing the user to juggle extra JSONL error files.

Recommended workflow:

```bash
# Generate or resume
uv run cli.py generate -c config.yaml

# Inspect the current dataset
uv run cli.py qa datasets/agentic_dataset.jsonl

# Apply cleanup rules if needed
uv run cli.py cleanup datasets/agentic_dataset.jsonl --max-failed-tool-calls 0 --yes

# Resume again; rows not present in the cleaned dataset are retried automatically
uv run cli.py generate -c config.yaml
```

## Architecture

```text
.
├── LICENSE
├── README.md
├── cli.py                        # Thin wrapper for the unified CLI
├── config.example.yaml           # Example configuration
├── requirements.txt
└── src/
    ├── agentic_datagen/
    │   ├── cli.py                # Unified generate / qa / cleanup CLI
    │   ├── generator.py          # Main orchestrator
    │   ├── agent_session.py      # Session management
    │   ├── tool_registry.py      # Built-ins, Python tools, and MCP tools
    │   ├── tools.py              # ToolRegistry compatibility layer
    │   ├── run_manifest.py       # Per-prompt status and attempt tracking
    │   ├── formatter.py          # Dataset formatting and training-safe checks
    │   ├── dataset_qa.py         # QA reporting and final dataset totals
    │   ├── dataset_cleanup.py    # Backup-first cleanup workflow
    │   └── utils.py              # Prompt loading utilities
    ├── custom_tools/
    │   └── example_tools.py      # Example custom Python tools
    ├── docker/
    │   └── session-runtime.Dockerfile
    └── tests/
        └── test_infrastructure.py
```

## Contributing

This tool is designed to be extensible:

- Add new built-ins in `src/agentic_datagen/tool_registry.py`
- Add pluggable Python tools under `src/custom_tools/`
- Connect MCP HTTP servers through `config.yaml`
- Modify formatting in `src/agentic_datagen/formatter.py`
- Extend session logic in `src/agentic_datagen/agent_session.py`

## Troubleshooting

### Context Length Exceeded

If the LLM provider returns a context length error, the session is marked `fatal_error`. This happens when the model generates very large outputs without completing. Consider:

- Setting a lower `model.max_turns` limit in config
- Using a model with larger context window
- Breaking complex prompts into smaller tasks

### Workspace Cleanup Failures

Docker containers may create files with race conditions (e.g., npm cache). The generator now retries cleanup with `rm -rf` fallback. If workspaces persist after runs, clean them manually:

```bash
rm -rf sandbox/session_*
```

### Empty LLM Responses

Sessions with `LLM call failed: empty response choices` are marked `retryable_error`. These are typically transient provider issues and will succeed on resume.

### Docker Tool Execution

When using `tool_scope: all`, ensure the Docker image has all required tools (Python, Node, etc.). The included `session-runtime.Dockerfile` provides a complete environment.

## License

[MIT](https://github.com/TeichAI/agentic-datagen/blob/main/LICENSE)

---

This tool was created by TeichAI.

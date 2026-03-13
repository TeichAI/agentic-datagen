# Agentic Dataset Generator

A tool for creating agentic coding datasets with tool-calling capabilities.

## Overview

This tool generates synthetic agentic datasets by:

1. Loading prompts from a configured source
2. Creating isolated workspaces for each prompt
3. Running an AI agent with Windsurf/Cursor/Codex-like capabilities (file operations, code search, etc.)
4. Recording all reasoning, tool calls, and responses
5. Validating and appending to a JSONL dataset file

## Features

- **Windsurf/Cursor/Codex-like Tools**: File operations (read, write, edit), directory listing, code search, command execution.
- **Extensible Tool Registry**: Built-in tools, custom Python tools, and MCP-backed HTTP tools can all be enabled from config.
- **Web Search**: Live integration with SearXNG instances.
- **Live Metrics & Progress**: Real-time CLI tracking of cost (USD), token count, and completion status via `tqdm`.
- **Workspace Isolation**: Each prompt gets its own workspace directory (`sandbox/` by default).
- **Session Recording**: Complete multi-turn trajectories including reasoning and tool outputs.
- **Resume Support**: Automatically skips already processed prompts.
- **Run Manifest**: Per-prompt manifest with status, attempts, workspace path, usage, and output routing.
- **Error Capture & Retry**: Optionally route failed sessions to a dedicated JSONL file for retries.
- **Flexible Prompt Sources**: Accepts `.txt`, `.json`, and `.jsonl` sources.

## Installation

```bash
# Clone the repository
git clone https://github.com/TeichAI/agentic_datagen.git
cd agentic_datagen

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```bash
# Create config from example
cp config.example.yaml config.yaml

# Run generation
python cli.py -c config.yaml
```

## Configuration

The tool uses a simple YAML configuration file. See `config.example.yaml` for a template, and `config.errors.yaml` for an error-retry template.

### Minimal Configuration

```yaml
api:
  model: "anthropic/claude-3.5-sonnet"
  api_key: "your-api-key"
  searxng_url: "http://localhost:your-searxng-port"

prompts:
  source: "prompts.txt" # .txt, .jsonl, or .json

workspace:
  base_dir: "sandbox"

agent:
  tools_enabled:
    - read_file
    - write_file
    - run_command
    - web_search

output:
  dataset_file: "datasets/agentic_dataset.jsonl"
  error_dataset_file: "datasets/agentic_dataset_errors.jsonl"

processing:
  concurrency: 10
  resume: true
```

### API Options

```yaml
api:
  provider: "openrouter" # Provider name (optional)
  base_url: "https://openrouter.ai/api/v1/chat/completions" # Override API endpoint
  api_key_env: "OPENROUTER_API_KEY" # Read API key from env instead of api_key
  reasoning_effort: "medium" # Optional: OpenRouter reasoning effort (low|medium|high)
  max_retries: 5 # Retries for retryable transport/provider failures
  backoff_base_seconds: 2.0 # Exponential backoff base delay
  backoff_max_seconds: 60.0 # Maximum retry delay
  timeout: 120 # Request timeout in seconds
```

### Prompt Sources

Supported formats: `.txt`, `.json`, `.jsonl`.

- **Text**: each line is a prompt.
- **JSON/JSONL**: each object can use one of these keys: `prompt`, `input`, `question`, `task`, `query`.

### Output Files

```yaml
output:
  dataset_file: "datasets/agentic_dataset.jsonl"
  error_dataset_file: "datasets/agentic_dataset_errors.jsonl" # Optional
  run_manifest_file: "datasets/agentic_dataset.manifest.json" # Optional
  append_mode: true
```

- `dataset_file` stores successful sessions.
- `error_dataset_file` (optional) stores failed sessions with `metadata.error` and full `usage` so you can retry later.
- `run_manifest_file` stores one record per prompt with status, attempts, route, and usage metadata.
- Set `error_dataset_file` to `null`/omit it if you don’t want a separate error file.
- When retrying, **never** write errors back into the same file you’re using as the prompt source.

### Agent Prompting

If `agent.system_prompt` is omitted, the generator uses a short default prompt tuned for code-editing trajectories:

```text
You are a coding agent. Use tools deliberately, inspect before editing, and finish the user's request with working files inside the workspace. When Context7 documentation tools are available and you are working with libraries or frameworks, use Context7 to fetch the latest relevant docs before making library-specific changes.
```

You can override it in config when you want a stricter or domain-specific behavior.

## Usage

```bash
# Run with config file
python cli.py -c config.yaml
```

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

1. Create a Python module, for example `custom_tools/example_tools.py`.
2. Export either:
   - `TOOLS`: a list of tool spec dictionaries, or
   - `register_tools(registry)`: a function that returns tool specs or registers them directly.
3. Add the module path to `tools.custom_python_modules`.
4. Add the tool name to `agent.tools_enabled`.

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

agent:
  tools_enabled:
    - context7:*
```

Notes:

- Remote MCP tools are exposed locally as either `mcp__<server>__<tool>` or `<tool_name_prefix>__<tool>`.
- A selector like `context7:*` enables every discovered tool from that MCP server, which is convenient for Context7.
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

The generator can write failed sessions to a dedicated JSONL file so you can retry them later without mixing with successful entries.

### Initial run with error capture

```yaml
output:
  dataset_file: datasets/agentic_dataset.jsonl
  error_dataset_file: datasets/agentic_dataset_errors.jsonl
processing:
  resume: false
```

### Retry only failed prompts

Use the previous **error dataset** as the prompt source, and write new failures to a **different** error file. This prevents the retry from appending back into the same file you are reading.

```yaml
output:
  dataset_file: datasets/agentic_dataset.jsonl
  error_dataset_file: datasets/agentic_dataset_errors_retry.jsonl
prompts:
  source: datasets/agentic_dataset_errors.jsonl
  limit: 0
processing:
  resume: false
```

When the retry succeeds, entries are appended to `dataset_file`. Any remaining failures go to `error_dataset_file`.

## Architecture

```text
.
├── cli.py              # CLI entry point
├── generator.py        # Main orchestrator
├── agent_session.py    # Session management
├── tool_registry.py    # Extensible tool registry for built-ins, Python tools, and MCP tools
├── tools.py            # Compatibility import wrapper for ToolRegistry
├── run_manifest.py     # Per-prompt status and attempt tracking
├── custom_tools/       # Example custom tool modules
├── formatter.py        # OpenAI format converter
├── utils.py            # Prompt loading utilities
├── config.example.yaml # Example configuration
└── README.md           # This file
```

## Contributing

This tool is designed to be extensible:

- Add new built-ins in `tool_registry.py`
- Add pluggable Python tools under `custom_tools/`
- Connect MCP HTTP servers through `config.yaml`
- Modify formatting in `formatter.py`
- Extend session logic in `agent_session.py`

## License

[MIT](https://github.com/TeichAI/agentic-datagen/blob/main/LICENSE)

---

This tool was created by TeichAI.

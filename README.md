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
- **Web Search**: Live integration with SearXNG instances.
- **Live Metrics & Progress**: Real-time CLI tracking of cost (USD), token count, and completion status via `tqdm`.
- **Workspace Isolation**: Each prompt gets its own workspace directory (`sandbox/` by default).
- **Session Recording**: Complete multi-turn trajectories including reasoning and tool outputs.
- **Resume Support**: Automatically skips already processed prompts.
- **Error Capture & Retry**: Optionally route failed sessions to a dedicated JSONL file for retries.
- **Flexible Prompt Sources**: Accepts `.txt`, `.json`, and `.jsonl` sources (including `query` fields).

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
  searxng_url: "https://searxng.gptbox.dev"

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
  append_mode: true
```

- `dataset_file` stores successful sessions.
- `error_dataset_file` (optional) stores failed sessions with `metadata.error` and full `usage` so you can retry later.
- Set `error_dataset_file` to `null`/omit it if you don’t want a separate error file.
- When retrying, **never** write errors back into the same file you’re using as the prompt source.

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
├── tools.py            # Tool registry and implementations
├── formatter.py        # OpenAI format converter
├── utils.py            # Prompt loading utilities
├── config.example.yaml # Example configuration
└── README.md           # This file
```

## Contributing

This tool is designed to be extensible:

- Add new tools in `tools.py`
- Modify formatting in `formatter.py`
- Extend session logic in `agent_session.py`

## License

[Apache 2.0](https://github.com/TeichAI/agentic-datagen/blob/main/LICENSE)

---

*This tool was created by TeichAI*

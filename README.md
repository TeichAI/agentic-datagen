# Agentic Dataset Generator

A tool for creating agentic coding datasets with tool-calling capabilities, compatible with the Nvidia Nemotron-Agentic-v1 format.

## Overview

This tool generates synthetic agentic datasets by:

1. Loading prompts from a configured source
2. Creating isolated workspaces for each prompt
3. Running an AI agent with Windsurf/Cursor/Codex-like capabilities (file operations, code search, etc.)
4. Recording all reasoning, tool calls, and responses
5. Formatting output to match Nemotron-Agentic-v1 structure
6. Validating and appending to a JSONL dataset file

## Features

- **Windsurf/Cursor/Codex-like Tools**: File operations (read, write, edit), directory listing, code search, command execution.
- **Web Search**: Live integration with SearXNG instances.
- **Live Metrics & Progress**: Real-time CLI tracking of cost (USD), token count, and completion status via `tqdm`.
- **Workspace Isolation**: Each prompt gets its own workspace directory (`sandbox/` by default).
- **Session Recording**: Complete multi-turn trajectories including reasoning and tool outputs.
- **Nemotron Format**: Native compatibility with Nvidia Nemotron-Agentic-v1 dataset structure.
- **Resume Support**: Automatically skips already processed prompts.

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/agentic_datagen.git
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

The tool uses a simple YAML configuration file. See `config.example.yaml` for a template.

### Minimal Configuration

```yaml
api:
  model: "anthropic/claude-3.5-sonnet"
  api_key: "your-api-key"
  searxng_url: "https://searxng.gptbox.dev"

prompts:
  source: "prompts.txt"

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
```

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
5. Formatting output to match Nemotron-Agentic-v1 structure
6. Validating and appending to a JSONL dataset file
7. Cleaning up workspaces (if configured)

## Architecture

```text
.
├── cli.py              # CLI entry point
├── generator.py        # Main orchestrator
├── agent_session.py    # Session management
├── tools.py            # Tool registry and implementations
├── formatter.py        # Nemotron format converter
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

Same as parent TeichAI project.

---

*This tool was created by TeichAI*

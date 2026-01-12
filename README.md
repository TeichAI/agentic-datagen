# Agentic Dataset Generator

A tool for creating agentic coding datasets with tool-calling capabilities, compatible with the Nvidia Nemotron-Agentic-v1 format.

## Overview

This tool generates synthetic agentic datasets by:
1. Loading prompts from a configured source
2. Creating isolated workspaces for each prompt
3. Running an AI agent with codex-like capabilities (file operations, code search, etc.)
4. Recording all reasoning, tool calls, and responses
5. Formatting output to match Nemotron-Agentic-v1 structure
6. Validating and appending to a JSONL dataset file

## Quick Start

```bash
# Set up environment
export OPENROUTER_API_KEY="your-api-key"

# Install dependencies
pip install -r requirements.txt

# Create config from example
cp agentic_datagen/config.example.yaml config.yaml

# Run generation
python -m agentic_datagen.cli -c config.yaml
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

## Features

- **Codex-like Tools**: File operations (read, write, edit), directory listing, code search, command execution.
- **Web Search**: Live integration with SearXNG instances.
- **Live Metrics & Progress**: Real-time CLI tracking of cost (USD), token count, and completion status via `tqdm`.
- **Workspace Isolation**: Each prompt gets its own workspace directory (`sandbox/` by default).
- **Session Recording**: Complete multi-turn trajectories including reasoning and tool outputs.
- **Nemotron Format**: Native compatibility with Nvidia Nemotron-Agentic-v1 dataset structure.
- **Resume Support**: Automatically skips already processed prompts.

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Configuration

Create a configuration file (see `config.example.yaml`):

```yaml
# API Configuration
api:
  provider: "openrouter"
  base_url: "https://openrouter.ai/api/v1/chat/completions"
  model: "anthropic/claude-3.5-sonnet"
  api_key_env: "OPENROUTER_API_KEY"

# Prompts Configuration
prompts:
  source: "combined.txt"
  limit: null  # Process all prompts
  shuffle: false

# Workspace Configuration
workspace:
  base_dir: "agentic_workspaces"
  cleanup: true  # Delete workspace after success
  preserve_on_error: true

# Agent Configuration
agent:
  system_prompt: |
    You are a helpful coding assistant with access to file operations and code analysis tools.
    Complete the user's task thoroughly and efficiently.
  max_turns: 50
  tools_enabled:
    - read_file
    - write_file
    - edit_file
    - list_directory
    - search_code
    - run_command

# Output Configuration
output:
  dataset_file: "datasets/agentic_dataset.jsonl"
  format: "nemotron"
  validate: true
  append_mode: true

# Processing Configuration
processing:
  concurrency: 1  # Start with 1 for safety
  resume: true
```

## Usage

```bash
# Run with config file
python -m agentic_datagen.cli -c config.yaml
```

## Live Metrics & Progress

The tool provides a live CLI progress bar using `tqdm`, tracking:
- **Total Cost**: Real-time USD spend (based on OpenRouter/API usage reporting).
- **Token Count**: Total cumulative input and output tokens.
- **Completion Rate**: Remaining prompts and estimated time to completion.

## Available Tools

- **read_file**: Read file contents from workspace
- **write_file**: Write content to a file
- **edit_file**: Replace text in a file
- **list_directory**: List files and directories
- **search_code**: Search for patterns in files
- **run_command**: Execute shell commands (with timeout)
- **web_search**: Placeholder for web search integration

## Workflow

1. **Load Prompts**: Reads prompts from configured source
2. **Create Workspace**: Creates isolated directory for each prompt
3. **Run Agent**: Executes agent with tool access
4. **Record Session**: Captures all interactions and tool calls
5. **Format Output**: Converts to Nemotron-compatible format
6. **Validate**: Ensures proper structure
7. **Save**: Appends to dataset file
8. **Cleanup**: Optionally removes workspace

## Advanced Features

### Resume Capability

The tool automatically detects completed prompts and skips them:

```yaml
processing:
  resume: true
```

### Concurrent Processing

Process multiple prompts in parallel:

```yaml
processing:
  concurrency: 5  # Process 5 prompts at once
```

### Workspace Preservation

Keep workspaces for debugging:

```yaml
workspace:
  cleanup: false  # Keep all workspaces
  preserve_on_error: true  # Keep only failed ones
```

### Custom System Prompts

Tailor the agent's behavior:

```yaml
agent:
  system_prompt: |
    You are an expert Python developer specializing in data science.
    Use best practices and write clean, documented code.
```

## Troubleshooting

### API Errors

- Check your API key is set correctly
- Verify the model supports tool calling
- Check rate limits and quotas

### Workspace Issues

- Ensure sufficient disk space
- Check file permissions
- Review preserved workspaces for errors

### Validation Failures

- Check logs for specific validation errors
- Review the generated entries
- Ensure tools are returning proper formats

## Examples

### Basic Usage

```bash
# Create config
cp agentic_datagen/config.example.yaml my_config.yaml

# Edit config with your settings
# Set API key
export OPENROUTER_API_KEY="sk-..."

# Run
python -m agentic_datagen.cli -c my_config.yaml
```

### Process First 10 Prompts

```yaml
prompts:
  source: "combined.txt"
  limit: 10
```

### Use Different Model

```yaml
api:
  model: "openai/gpt-4"
```

## Architecture

```
agentic_datagen/
├── __init__.py
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

*Generated by TeichAI Agentic Dataset Tooling*


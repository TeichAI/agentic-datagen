# Agentic Dataset Generator - Architecture

## Overview

This tool creates agentic coding datasets by simulating a coding assistant with tool-calling capabilities, similar to how Codex or GitHub Copilot work, but recording the entire interaction trajectory for training purposes.

## Core Components

### 1. Generator (`generator.py`)

**Main orchestrator** that manages the entire dataset generation pipeline.

**Key responsibilities:**
- Load configuration from YAML
- Initialize logging and API connections
- Load prompts from various sources
- Track completed prompts for resume capability
- Create and manage workspaces
- Process prompts sequentially or concurrently
- Validate and save dataset entries

**Flow:**
```
Load Config → Load Prompts → Filter Completed → Process Each Prompt → Validate → Save → Cleanup
```

### 2. Agent Session (`agent_session.py`)

**Session manager** for individual prompt processing.

**Key responsibilities:**
- Initialize workspace and tools
- Manage conversation with LLM
- Handle tool calls and responses
- Track conversation history
- Return complete session data

**Flow:**
```
Create Session → Send Prompt → LLM Response → Tool Calls? → Execute Tools → Continue → Final Response
```

### 3. Tool Registry (`tools.py`)

**Tool implementation** providing codex-like capabilities.

**Available tools:**
- `read_file`: Read workspace files
- `write_file`: Create/overwrite files
- `edit_file`: Find-and-replace in files
- `list_directory`: Browse workspace
- `search_code`: Grep-like search
- `run_command`: Execute shell commands
- `web_search`: Placeholder for web integration

**Security:**
- All file operations are sandboxed to workspace
- Path traversal protection
- Command timeout (30s)

### 4. Formatter (`formatter.py`)

**Responsibilities:**
- Convert session data to the proper format
- Validate entry structure
- Generate JSONL lines

**Output structure:**
```json
{
  "messages": [...],  // Full conversation with tool calls
  "metadata": {       // Session metadata
    "session_id": "...",
    "prompt": "...",
    "turns": N,
    "completed": true/false,
    "tool_calls_count": N
  }
}
```

### 5. Utils (`utils.py`)

**Prompt loading utilities** supporting multiple formats.

**Supported sources:**
- `.txt` files (one prompt per line)
- `.jsonl` files (extracts from messages/prompt fields)
- `.json` files (same as JSONL)
- `.md` files (single prompt)
- Directories of `.md` files (numbered prompts)

## Data Flow

```
┌─────────────┐
│   Config    │
│   (YAML)    │
└──────┬──────┘
       │
       v
┌─────────────────┐
│   Generator     │
│  - Load prompts │
│  - Filter done  │
└──────┬──────────┘
       │
       v
┌─────────────────────┐
│  For each prompt:   │
│                     │
│  ┌───────────────┐  │
│  │ Create        │  │
│  │ Workspace     │  │
│  └───────┬───────┘  │
│          │          │
│          v          │
│  ┌───────────────┐  │
│  │ Agent Session │  │
│  │ - LLM calls   │  │
│  │ - Tool exec   │  │
│  └───────┬───────┘  │
│          │          │
│          v          │
│  ┌───────────────┐  │
│  │ Format        │  │
│  └───────┬───────┘  │
│          │          │
│          v          │
│  ┌───────────────┐  │
│  │ Validate      │  │
│  └───────┬───────┘  │
│          │          │
│          v          │
│  ┌───────────────┐  │
│  │ Append to     │  │
│  │ Dataset       │  │
│  └───────┬───────┘  │
│          │          │
│          v          │
│  ┌───────────────┐  │
│  │ Cleanup       │  │
│  │ Workspace     │  │
│  └───────────────┘  │
└─────────────────────┘
```

## Workspace Management

Each prompt gets an isolated workspace:

```
agentic_workspaces/
├── session_000001/
│   ├── file1.py
│   ├── file2.txt
│   └── ...
├── session_000002/
│   └── ...
└── ...
```

**Lifecycle:**
1. Create before session
2. Agent can read/write/edit files
3. Cleanup after success (configurable)
4. Preserve on error (configurable)

## Configuration System

YAML-based configuration with sections:

- **api**: LLM provider settings
- **prompts**: Source and filtering
- **workspace**: Directory management
- **agent**: System prompt and tools
- **output**: Dataset file settings
- **processing**: Concurrency and resume
- **logging**: Verbosity and files

## Resume Capability

The generator tracks completed prompts by:
1. Reading existing dataset file
2. Extracting prompts from metadata
3. Skipping already-processed prompts
4. Continuing from where it left off

This allows:
- Crash recovery
- Incremental processing
- Cost management

## Concurrency Model

**Sequential (concurrency: 1):**
- Process one prompt at a time
- Easier debugging
- Lower memory usage

**Parallel (concurrency: N):**
- Process N prompts simultaneously
- Faster completion
- Higher resource usage
- ThreadPoolExecutor-based

## Error Handling

**Levels:**
1. **Tool errors**: Returned to LLM as tool results
2. **Session errors**: Logged, workspace preserved
3. **Fatal errors**: Stop processing, report to user

**Recovery:**
- Resume from last successful prompt
- Preserved workspaces for debugging
- Detailed logging

## Token & Cost Tracking

The system automatically monitors API usage:

- **Per-Session Tracking**: Each `AgentSession` records its own token usage and cost.
- **Global Aggregation**: `AgenticDatasetGenerator` aggregates metrics across all sessions.
- **Live Feedback**: Real-time stats are piped to the CLI progress bar (`tqdm`).

## Format Reference

Each entry includes:

- `messages`: Full multi-turn conversation.
- `tools`: List of available tool definitions in OpenAI format.
- `metadata`: Session stats and completion status.
- `usage`: Token and cost breakdown for that specific trajectory.

## Extension Points

**Add new tools:**
```python
# In tools.py
def my_tool(self, arg1: str, arg2: int) -> str:
    # Implementation
    return result

# Add to registry
self.tools["my_tool"] = self.my_tool

# Add definition in get_tool_definitions()
```

**Custom formatters:**
```python
# In formatter.py
class CustomFormatter:
    @staticmethod
    def format_session(session_data):
        # Custom formatting
        return formatted_data
```

**Different LLM providers:**
```python
# In agent_session.py
def _call_llm(self, messages, enabled_tools):
    # Custom API integration
    return response
```

## Performance Considerations

**Memory:**
- Each workspace is isolated
- Concurrent sessions multiply memory usage
- Large conversations consume more memory

**Disk:**
- Workspaces can accumulate if not cleaned
- Dataset file grows linearly
- Logs can become large

**API:**
- Rate limits depend on provider
- Costs scale with prompt count and turns
- Retries help with transient errors

**Optimization tips:**
1. Start with low concurrency
2. Enable workspace cleanup
3. Use prompt limits for testing
4. Monitor API costs
5. Validate early and often

## Security Considerations

**Workspace isolation:**
- All file operations are sandboxed
- Path traversal protection
- No access outside workspace

**Command execution:**
- 30-second timeout
- Runs in workspace directory
- Captures stdout/stderr

**API keys:**
- Loaded from environment
- Never logged in full
- Support for .env files

## Comparison to nvidia/Nemotron-Agentic-v1

**Similarities:**
- Multi-turn conversations
- Tool calling support
- JSONL format
- Metadata tracking

**Differences:**
- We use isolated workspaces
- Single-agent (not multi-role simulation)
- Focused on coding tasks
- Configurable tool set

## Future Enhancements

Potential improvements:
- Multi-agent simulation (user/assistant/judge)
- Web search integration
- Git operations
- Package installation
- Test execution
- Code quality checks
- Automatic validation
- HuggingFace upload integration

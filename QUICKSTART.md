# Quick Start Guide

## 1. Install Dependencies

```bash
pip install -r requirements.txt
```

## 2. Set API Key

You can provide your API key directly in the configuration file or via environment variables.

### Option A: Direct Configuration (Recommended)

Add it to your `config.yaml`:

```yaml
api:
  api_key: "your-api-key-here"
```

### Option B: Environment Variables

```bash
# Windows PowerShell
$env:OPENROUTER_API_KEY="your-api-key-here"

# Linux/Mac
export OPENROUTER_API_KEY="your-api-key-here"
```

## 3. Configure SearXNG (Optional)

If using the `web_search` tool, provide your SearXNG URL in the config:

```yaml
api:
  searxng_url: "https://searxng.gptbox.dev"
```

## 4. Run Generation

```bash
python cli.py -c config.yaml
```

## 5. Live Monitoring

The CLI provides a live progress bar tracking:

- **Total Cost**: Cumulative USD spend.
- **Tokens**: Total input + output tokens.
- **Progress**: Remaining prompts and ETA.

## 6. Full Dataset Generation

```bash
python cli.py -c config.yaml
```

## Common Configurations

### Process All Prompts

```yaml
prompts:
  source: "combined.txt"
  limit: null
```

### Use Different Model

```yaml
api:
  model: "openai/gpt-4o"
  # or
  model: "google/gemini-pro-1.5"
  # or
  model: "anthropic/claude-3-opus"
```

### Enable Concurrency

```yaml
processing:
  concurrency: 5  # Process 5 prompts in parallel
```

### Clean Up Workspaces

```yaml
workspace:
  cleanup: true  # Delete workspaces after success
  preserve_on_error: true  # Keep failed ones for debugging
```

### Disable Resume

```yaml
processing:
  resume: false  # Start from scratch
```

## Troubleshooting

### "Missing API key"

- Ensure `OPENROUTER_API_KEY` is set in environment or `.env` file
- Check the key is valid and has credits

### "Module not found"

- Ensure you are in the repository root
- Ensure dependencies are installed: `pip install -r requirements.txt`

### "Permission denied"

- Check file permissions on output directory
- Ensure workspace directory is writable

### Tool Execution Fails

- Check workspace is created correctly
- Review logs for specific errors
- Inspect preserved workspaces

### Validation Errors

- Check the log file for details
- Review the session data structure
- Ensure tools return proper JSON

## Next Steps

1. **Review Generated Data**: Inspect the JSONL output to ensure quality
2. **Adjust System Prompt**: Tailor agent behavior for your use case
3. **Scale Up**: Increase concurrency for faster processing
4. **Monitor Progress**: Watch logs and intermediate outputs
5. **Upload to HuggingFace**: Share your dataset with the community

## Example Output Structure

Each line in the JSONL file contains:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful coding assistant..."
    },
    {
      "role": "user",
      "content": "Create a Python script that..."
    },
    {
      "role": "assistant",
      "content": "I'll help you create that script.",
      "tool_calls": [
        {
          "id": "call_1",
          "type": "function",
          "function": {
            "name": "write_file",
            "arguments": "{\"file_path\": \"script.py\", \"content\": \"...\"}"
          }
        }
      ]
    },
    {
      "role": "tool",
      "tool_call_id": "call_1",
      "name": "write_file",
      "content": "{\"success\": true, \"result\": \"Successfully wrote...\"}"
    }
  ],
  "metadata": {
    "session_id": "session_000001",
    "prompt": "Create a Python script that...",
    "turns": 3,
    "completed": true,
    "tool_calls_count": 1
  }
}
```

## Tips for Best Results

1. **Start Small**: Test with `limit: 10` before processing thousands
2. **Monitor Costs**: Track API usage and costs
3. **Preserve Errors**: Keep failed workspaces to debug issues
4. **Validate Output**: Regularly check dataset quality
5. **Backup Data**: Keep copies of generated datasets
6. **Iterate**: Refine system prompts based on results

# Configuration Directory

This directory contains configuration files for the VisFly-Eureka project.

## API Keys Configuration

### Security Notice
**NEVER commit API keys to version control!**

### Setup Instructions

1. Copy the example file:
   ```bash
   cp api_keys.example.yaml api_keys.yaml
   ```

2. Edit `api_keys.yaml` and add your actual API keys:
   ```yaml
   openai:
     api_key: "your-actual-openai-key"
   
   yunwu:
     api_key: "your-actual-yunwu-key"
     base_url: "https://yunwu.ai/v1"
   ```

3. Alternative: Use environment variables:
   ```bash
   export OPENAI_API_KEY="your-key-here"
   export YUNWU_API_KEY="your-key-here"
   export ANTHROPIC_API_KEY="your-key-here"
   ```

### Files

- `api_keys.example.yaml` - Example template (safe to commit)
- `api_keys.yaml` - Your actual API keys (gitignored, never commit)
- `llm/` - LLM provider configurations

### Important Notes

- The `api_keys.yaml` file is automatically ignored by git
- If you accidentally commit API keys, immediately:
  1. Revoke the exposed keys
  2. Generate new keys
  3. Remove the commit from history using `git filter-branch` or BFG Repo-Cleaner
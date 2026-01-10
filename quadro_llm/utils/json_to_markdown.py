"""
Tool to convert LLM conversation JSON files to readable Markdown format.

This tool converts the JSON conversation files saved by LLMEngine.save_conversations()
into well-formatted Markdown documents for easier reading and analysis.

Usage (Command Line):
    python quadro_llm/utils/json_to_markdown.py <json_file> [output_file]
    
    Examples:
        # Convert to Markdown and print to stdout
        python quadro_llm/utils/json_to_markdown.py results/.../llm_conversations_iteration_0.json
        
        # Convert and save to file
        python quadro_llm/utils/json_to_markdown.py results/.../llm_conversations_iteration_0.json output.md
        
        # Automatically detect iteration number from filename
        python quadro_llm/utils/json_to_markdown.py results/.../llm_conversations_iteration_4.json

Usage (Python):
    from quadro_llm.utils.json_to_markdown import json_to_markdown
    import json
    
    with open('llm_conversations_iteration_0.json', 'r') as f:
        data = json.load(f)
    
    markdown = json_to_markdown(data, iteration=0)
    print(markdown)
    
Output Format:
    - Metadata (iteration, timestamp, statistics)
    - Task description
    - Feedback from previous iteration
    - Conversation messages (system prompt, user request)
    - Generated reward functions (formatted as code blocks)
    - Model configuration
    - Token usage statistics
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any


def format_code_block(code: str, language: str = "python") -> str:
    """Format code string as markdown code block"""
    return f"```{language}\n{code.strip()}\n```"


def format_message(message: Dict[str, str]) -> str:
    """Format a single message (system/user/assistant) as markdown"""
    role = message.get("role", "unknown")
    content = message.get("content", "")
    
    role_display = {
        "system": "System Prompt",
        "user": "User Request",
        "assistant": "Assistant Response",
    }.get(role, role.capitalize())
    
    lines = [f"### {role_display}", ""]
    
    # System prompts should be plain text (instructions), not code
    if role == "system":
        lines.append(content)
    # Check if content contains code blocks (preserve existing formatting)
    elif "```" in content:
        # Content already has code blocks, preserve them
        lines.append(content)
    elif role == "user" and ("```python" in content or "```text" in content or "```" in content):
        # User message already has code blocks, preserve as-is
        lines.append(content)
    elif role == "user" and ("def get_reward" in content or "class " in content or "import " in content):
        # User message contains code, but check if it's embedded in text
        # If it starts with "The Python environment is:" or similar, preserve the text structure
        if content.count("```") >= 2:
            # Already has code blocks
            lines.append(content)
        else:
            # Extract code sections and format them
            # This is complex, so just preserve as-is for user messages
            lines.append(content)
    else:
        # Regular text, preserve line breaks
        lines.append(content)
    
    lines.append("")
    return "\n".join(lines)


def json_to_markdown(json_data: List[Dict[str, Any]], iteration: int = None) -> str:
    """Convert LLM conversation JSON to Markdown format"""
    
    markdown_lines = []
    
    # Handle both single conversation object and list of conversations
    if isinstance(json_data, dict):
        conversations = [json_data]
    else:
        conversations = json_data
    
    for conv_idx, conv in enumerate(conversations):
        # Header
        if len(conversations) > 1:
            markdown_lines.append(f"# Conversation {conv_idx + 1}")
            markdown_lines.append("")
        else:
            markdown_lines.append("# LLM Conversation")
            markdown_lines.append("")
        
        # Metadata
        timestamp = conv.get("timestamp", "Unknown")
        if iteration is not None:
            markdown_lines.append(f"**Iteration:** {iteration}  ")
        markdown_lines.append(f"**Timestamp:** {timestamp}")
        markdown_lines.append("")
        
        # Task Description
        task_desc = conv.get("task_description", "")
        if task_desc:
            markdown_lines.append("## Task Description")
            markdown_lines.append("")
            markdown_lines.append(task_desc.strip())
            markdown_lines.append("")
        
        # Feedback
        feedback = conv.get("feedback", "")
        if feedback and feedback.strip() != "This is the first iteration. Focus on basic task completion.":
            markdown_lines.append("## Feedback from Previous Iteration")
            markdown_lines.append("")
            markdown_lines.append(feedback.strip())
            markdown_lines.append("")
        
        # Generation Statistics
        samples_requested = conv.get("samples_requested", 0)
        samples_generated = conv.get("samples_generated", 0)
        if samples_requested > 0:
            markdown_lines.append("## Generation Statistics")
            markdown_lines.append("")
            markdown_lines.append(f"- **Samples Requested:** {samples_requested}")
            markdown_lines.append(f"- **Samples Generated:** {samples_generated}")
            markdown_lines.append("")
        
        # Messages (System + User prompts)
        messages = conv.get("messages", [])
        if messages:
            markdown_lines.append("## Conversation Messages")
            markdown_lines.append("")
            for msg in messages:
                markdown_lines.append(format_message(msg))
        
        # Generated Reward Functions
        results = conv.get("results", [])
        if results:
            markdown_lines.append("## Generated Reward Functions")
            markdown_lines.append("")
            markdown_lines.append(f"Generated {len(results)} reward function(s):")
            markdown_lines.append("")
            
            for idx, reward_code in enumerate(results):
                markdown_lines.append(f"### Reward Function {idx + 1}")
                markdown_lines.append("")
                markdown_lines.append(format_code_block(reward_code, "python"))
                markdown_lines.append("")
        
        # Model Configuration
        model_config = conv.get("model_config", {})
        if model_config:
            markdown_lines.append("## Model Configuration")
            markdown_lines.append("")
            for key, value in model_config.items():
                if value is not None:
                    markdown_lines.append(f"- **{key.replace('_', ' ').title()}:** {value}")
            markdown_lines.append("")
        
        # Token Usage
        token_usage = conv.get("token_usage", {})
        if token_usage:
            markdown_lines.append("## Token Usage")
            markdown_lines.append("")
            prompt_tokens = token_usage.get("prompt_tokens", 0)
            completion_tokens = token_usage.get("completion_tokens", 0)
            total_tokens = token_usage.get("total_tokens", 0)
            
            markdown_lines.append(f"- **Prompt Tokens:** {prompt_tokens:,}")
            markdown_lines.append(f"- **Completion Tokens:** {completion_tokens:,}")
            markdown_lines.append(f"- **Total Tokens:** {total_tokens:,}")
            
            if prompt_tokens > 0:
                completion_ratio = (completion_tokens / prompt_tokens) * 100
                markdown_lines.append(f"- **Completion/Prompt Ratio:** {completion_ratio:.1f}%")
            markdown_lines.append("")
        
        # Separator between multiple conversations
        if conv_idx < len(conversations) - 1:
            markdown_lines.append("---")
            markdown_lines.append("")
    
    return "\n".join(markdown_lines)


def main():
    """CLI entry point"""
    if len(sys.argv) < 2:
        print("Usage: python -m quadro_llm.utils.json_to_markdown <json_file> [output_file] [iteration]")
        print("Example: python -m quadro_llm.utils.json_to_markdown results/.../llm_conversations_iteration_0.json output.md 0")
        sys.exit(1)
    
    json_file = Path(sys.argv[1])
    output_file = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    iteration = int(sys.argv[3]) if len(sys.argv) > 3 else None
    
    # Try to extract iteration number from filename if not provided
    if iteration is None:
        filename = json_file.stem
        if "iteration_" in filename:
            try:
                iteration = int(filename.split("iteration_")[1])
            except (ValueError, IndexError):
                pass
    
    if not json_file.exists():
        print(f"Error: File not found: {json_file}")
        sys.exit(1)
    
    # Read JSON
    with open(json_file, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    # Convert to Markdown
    markdown_content = json_to_markdown(json_data, iteration=iteration)
    
    # Write output
    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        print(f"Markdown saved to: {output_file}")
    else:
        # Print to stdout
        print(markdown_content)


if __name__ == "__main__":
    main()

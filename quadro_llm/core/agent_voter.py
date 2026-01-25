"""Agent-based voter using Claude Agent SDK for selecting best reward function."""

import os
import asyncio
import json
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from omegaconf import DictConfig


@dataclass
class AgentVoterResult:
    """Result from agent voter"""
    selected_index: int
    selected_identifier: str
    reasoning: str
    confidence: float
    analysis_summary: str
    conversation: Optional[List[Dict]] = None


class AgentVoter:
    """Claude Agent SDK-based voter for selecting elite reward functions.
    
    Uses autonomous agent to explore training data, analyze trajectories,
    and make informed selection decisions.
    """
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._setup_environment()
    
    def _setup_environment(self):
        """Setup environment variables for Claude Agent SDK"""
        cfg = self.config
        
        # Debug: log config structure
        self.logger.debug(f"Agent config type: {type(cfg)}, keys: {list(cfg.keys()) if hasattr(cfg, 'keys') else 'N/A'}")
        
        # Handle nested "agent" key if present (Hydra config quirk)
        if hasattr(cfg, 'agent') and hasattr(cfg.agent, 'base_url'):
            self.logger.debug("Detected nested agent config, unwrapping")
            cfg = cfg.agent
        
        # API configuration
        base_url = cfg.get("base_url", "https://api.minimaxi.com/anthropic")
        os.environ["ANTHROPIC_BASE_URL"] = base_url
        os.environ["API_TIMEOUT_MS"] = str(cfg.get("timeout_ms", 3000000))
        os.environ["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"
        
        # API key - from config or environment
        api_key = cfg.get("api_key")
        if not api_key:
            api_key_env = cfg.get("api_key_env", "ANTHROPIC_AUTH_TOKEN")
            api_key = os.getenv(api_key_env)
        
        if api_key:
            os.environ["ANTHROPIC_AUTH_TOKEN"] = api_key
            self.logger.debug(f"API key set: {api_key[:10]}...{api_key[-4:]}")
        else:
            self.logger.warning("No API key found in config or environment!")
        
        # Model configuration
        model = cfg.get("model", "MiniMax-M2.1")
        os.environ["ANTHROPIC_MODEL"] = model
        os.environ["ANTHROPIC_SMALL_FAST_MODEL"] = model
        os.environ["ANTHROPIC_DEFAULT_SONNET_MODEL"] = model
        os.environ["ANTHROPIC_DEFAULT_OPUS_MODEL"] = model
        os.environ["ANTHROPIC_DEFAULT_HAIKU_MODEL"] = model
        
        self.model = model
        self.allowed_tools = ["Read", "Write", "Python", "Bash", "Glob"]
        self.logger.info(f"Agent voter configured with model: {model}")
    
    def vote(self, train_dir: Path, output_dir: Path, iteration: int, task_description: str) -> AgentVoterResult:
        """Run agent to analyze candidates and select the best one.
        
        Args:
            train_dir: Path to training results (e.g., results/xxx/train/iter0)
            output_dir: Path to save artifacts (e.g., results/xxx)
            iteration: Current iteration number
            task_description: Description of the task to evaluate against
            
        Returns:
            AgentVoterResult with selection and reasoning
        """
        return asyncio.run(self._vote_async(train_dir, output_dir, iteration, task_description))
    
    async def _vote_async(self, train_dir: Path, output_dir: Path, iteration: int, task_description: str) -> AgentVoterResult:
        """Async implementation of vote"""
        from claude_agent_sdk import (
            query, ClaudeAgentOptions,
            UserMessage, AssistantMessage, ResultMessage,
            TextBlock, ToolUseBlock, ToolResultBlock
        )
        
        # Output paths
        artifacts_dir = output_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        result_file = artifacts_dir / f"agent_voter_result_iter{iteration}.json"
        conv_file = artifacts_dir / f"agent_voter_conversation_iter{iteration}.json"
        
        # Count candidates
        candidates = sorted([d.name for d in train_dir.glob("sample*")],
                          key=lambda x: int(x.replace("sample", "")))
        
        self.logger.info(f"Agent voting on {len(candidates)} candidates from {train_dir}")
        
        prompt = f"""Select the best reward function candidate for the task: {task_description}.

Explore {train_dir} and analyze each sample's:
- result.json (metrics)
- trajectories/*.npz (trajectory data)
- tensorboard/ logs (training curves, optional)

Write your final decision to {result_file} as JSON:
{{"selected_index": <int>, "selected_identifier": "<sampleN>", "reasoning": "<why>", "confidence": <0-1>, "analysis_summary": "<overview>"}}"""

        options = ClaudeAgentOptions(
            model=self.model,
            allowed_tools=self.allowed_tools,
            permission_mode="acceptEdits",
            cwd=str(output_dir),
        )
        
        conversation = []
        
        # Don't break after ResultMessage - let the generator exhaust naturally
        # This avoids cancel scope cleanup errors from the SDK
        async for message in query(prompt=prompt, options=options):
            conversation.append(self._serialize_message(message))
            
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        self.logger.debug(f"Agent: {block.text[:200]}...")
            elif isinstance(message, ResultMessage):
                self.logger.info(f"Agent completed: {message.subtype}")
                # ResultMessage is the last message, loop will exit naturally
        
        # Save conversation
        with open(conv_file, 'w') as f:
            json.dump({"prompt": prompt, "messages": conversation}, f, indent=2, ensure_ascii=False)
        self.logger.info(f"Saved conversation to {conv_file}")
        
        # Read result
        with open(result_file) as f:
            result_data = json.load(f)
        
        return AgentVoterResult(
            selected_index=result_data["selected_index"],
            selected_identifier=result_data["selected_identifier"],
            reasoning=result_data["reasoning"],
            confidence=result_data.get("confidence", 0.8),
            analysis_summary=result_data.get("analysis_summary", ""),
            conversation=conversation
        )
    
    def _serialize_message(self, message) -> Dict[str, Any]:
        """Convert SDK message to JSON-serializable dict"""
        from claude_agent_sdk import (
            UserMessage, AssistantMessage, ResultMessage,
            TextBlock, ToolUseBlock, ToolResultBlock
        )
        
        if isinstance(message, UserMessage):
            content = []
            if isinstance(message.content, str):
                content = [{"type": "text", "text": message.content}]
            else:
                for block in message.content:
                    if isinstance(block, TextBlock):
                        content.append({"type": "text", "text": block.text})
                    elif isinstance(block, ToolResultBlock):
                        content.append({
                            "type": "tool_result",
                            "tool_use_id": block.tool_use_id,
                            "content": str(block.content)[:500]
                        })
            return {"role": "user", "content": content}
        
        elif isinstance(message, AssistantMessage):
            content = []
            for block in message.content:
                if isinstance(block, TextBlock):
                    content.append({"type": "text", "text": block.text})
                elif isinstance(block, ToolUseBlock):
                    content.append({
                        "type": "tool_use",
                        "name": block.name,
                        "id": block.id,
                        "input": block.input if hasattr(block, 'input') else {}
                    })
            return {"role": "assistant", "model": message.model, "content": content}
        
        elif isinstance(message, ResultMessage):
            return {
                "role": "result",
                "subtype": message.subtype,
                "total_cost_usd": getattr(message, 'total_cost_usd', None),
                "usage": getattr(message, 'usage', None)
            }
        
        return {"role": "unknown", "type": type(message).__name__}

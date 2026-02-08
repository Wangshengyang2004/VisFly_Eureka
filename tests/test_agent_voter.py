#!/usr/bin/env python3
"""
Test script for Claude Agent SDK-based Elite Voter.

Mimics the real pipeline behavior in agent_voter.py.

Usage:
    # Test with standalone implementation
    python tests/test_agent_voter.py --results-dir results/xxx --iteration 0
    
    # Test with real AgentVoter class
    python tests/test_agent_voter.py --results-dir results/xxx --iteration 0 --use-real-voter
    
    # Load config from YAML
    python tests/test_agent_voter.py --results-dir results/xxx --iteration 0 --config-file configs/agent/minimax.yaml
"""

import os
import sys
import asyncio
import argparse
import json
import logging
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Setup logging to match pipeline
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s][%(name)s][%(levelname)s] - %(message)s'
)
logger = logging.getLogger(__name__)


def load_agent_config(config_file: Path) -> Dict[str, Any]:
    """Load agent config from YAML file (mimics Hydra loading)"""
    with open(config_file) as f:
        raw_config = yaml.safe_load(f)
    
    # Handle nested "agent" key if present
    if "agent" in raw_config:
        config = raw_config["agent"]
        logger.debug(f"Loaded config with nested 'agent' key: {list(config.keys())}")
    else:
        config = raw_config
        logger.debug(f"Loaded flat config: {list(config.keys())}")
    
    return config


def setup_claude_env_from_config(config: Dict[str, Any]) -> str:
    """Setup environment variables from config dict (mimics AgentVoter._setup_environment)"""
    
    # API configuration
    base_url = config.get("base_url", "https://api.minimaxi.com/anthropic")
    os.environ["ANTHROPIC_BASE_URL"] = base_url
    os.environ["API_TIMEOUT_MS"] = str(config.get("timeout_ms", 3000000))
    os.environ["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"
    
    # API key - from config or environment
    api_key = config.get("api_key")
    if not api_key:
        api_key_env = config.get("api_key_env", "ANTHROPIC_AUTH_TOKEN")
        api_key = os.getenv(api_key_env)
    
    if api_key:
        os.environ["ANTHROPIC_AUTH_TOKEN"] = api_key
        logger.info(f"API key set: {api_key[:10]}...{api_key[-4:]}")
    else:
        logger.warning("No API key found in config or environment!")
    
    # Model configuration
    model = config.get("model", "MiniMax-M2.1")
    os.environ["ANTHROPIC_MODEL"] = model
    os.environ["ANTHROPIC_SMALL_FAST_MODEL"] = model
    os.environ["ANTHROPIC_DEFAULT_SONNET_MODEL"] = model
    os.environ["ANTHROPIC_DEFAULT_OPUS_MODEL"] = model
    os.environ["ANTHROPIC_DEFAULT_HAIKU_MODEL"] = model
    
    logger.info(f"Environment configured: base_url={base_url}, model={model}")
    return model


from claude_agent_sdk import (
    query, ClaudeAgentOptions, 
    UserMessage, AssistantMessage, ResultMessage,
    TextBlock, ToolUseBlock, ToolResultBlock
)


def serialize_message(message) -> Dict[str, Any]:
    """Convert SDK message to JSON-serializable dict"""
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


async def run_agent_voter(
    train_dir: Path, 
    output_dir: Path, 
    result_file: Path,
    conv_file: Path,
    model: str
) -> Dict[str, Any]:
    """
    Run agent voter (mimics AgentVoter._vote_async).
    
    Let generator exhaust naturally after ResultMessage - no break needed.
    """
    candidates = sorted([d.name for d in train_dir.glob("sample*")],
                       key=lambda x: int(x.replace("sample", "")))
    
    logger.info(f"Agent voting on {len(candidates)} candidates from {train_dir}")
    
    prompt = f"""Select the best reward function candidate for a drone flip task.

Perform a deep analysis for this task. Explore {train_dir} and analyze each sample using:
- result.json (metrics)
- trajectories/*.npz (trajectory data)
- tensorboard/ logs (training curves) — parse and use these for convergence and reward-curve comparison; include in your analysis.

Write your final decision to {result_file} as JSON:
{{"selected_index": <int>, "selected_identifier": "<sampleN>", "reasoning": "<why>", "analysis_summary": "<overview>"}}"""

    options = ClaudeAgentOptions(
        model=model,
        allowed_tools=["Read", "Write", "Python", "Bash", "Glob"],
        permission_mode="acceptEdits",
        cwd=str(output_dir),
    )
    
    conversation = []
    
    # Don't break after ResultMessage - let generator exhaust naturally
    # This avoids cancel scope cleanup errors from the SDK
    async for message in query(prompt=prompt, options=options):
        conversation.append(serialize_message(message))
        
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    logger.debug(f"Agent: {block.text[:200]}...")
        elif isinstance(message, ResultMessage):
            logger.info(f"Agent completed: {message.subtype}")
            # ResultMessage is the last message, loop will exit naturally
    
    logger.info("Async for loop exited naturally (generator exhausted)")
    
    # Save conversation
    with open(conv_file, 'w') as f:
        json.dump({"prompt": prompt, "messages": conversation}, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved conversation to {conv_file}")
    
    # Read result
    with open(result_file) as f:
        result_data = json.load(f)
    
    return result_data


def get_ground_truth(train_dir: Path) -> tuple:
    """Find best candidate by success_rate, then by final_distance"""
    best_candidate = None
    best_score = (-1, float('inf'))  # (success_rate, final_distance)
    
    for sample_dir in sorted(train_dir.glob("sample*"), 
                              key=lambda x: int(x.name.replace("sample", ""))):
        result_file = sample_dir / "result.json"
        if result_file.exists():
            with open(result_file) as f:
                data = json.load(f)
                success_rate = data.get("success_rate", 0)
                agg = data.get("aggregate_statistics", {})
                final_dist = agg.get("mean_final_distance", float('inf'))
                
                score = (success_rate, -final_dist)  # Higher success, lower distance is better
                if score > best_score:
                    best_score = score
                    best_candidate = sample_dir.name
    
    return best_candidate, best_score


def main():
    parser = argparse.ArgumentParser(description="Test Claude Agent Elite Voter")
    parser.add_argument("--results-dir", type=str, required=True,
                       help="Path to results directory (e.g., results/2026-01-25_19-19-47)")
    parser.add_argument("--iteration", type=int, default=0,
                       help="Iteration number to analyze")
    parser.add_argument("--output", type=str, default=None,
                       help="Output file path for result JSON")
    parser.add_argument("--config-file", type=str, default=None,
                       help="Path to agent config YAML file (e.g., configs/agent/minimax.yaml)")
    parser.add_argument("--api-key", type=str, default=None,
                       help="API key override (optional)")
    parser.add_argument("--use-real-voter", action="store_true",
                       help="Use real AgentVoter class")
    args = parser.parse_args()
    
    # Load config from YAML or use defaults
    if args.config_file:
        config_path = Path(args.config_file)
        if not config_path.is_absolute():
            config_path = PROJECT_ROOT / config_path
        
        if not config_path.exists():
            logger.error(f"Config file not found: {config_path}")
            return 1
        
        config = load_agent_config(config_path)
        logger.info(f"Loaded config from: {config_path}")
    else:
        # Default config
        config = {
            "base_url": "https://api.minimaxi.com/anthropic",
            "model": "MiniMax-M2.1",
            "api_key": args.api_key,  # May be None
            "timeout_ms": 3000000,
        }
        logger.info("Using default config")
    
    # Override API key if provided
    if args.api_key:
        config["api_key"] = args.api_key
    
    # Setup environment (mimics AgentVoter._setup_environment)
    model = setup_claude_env_from_config(config)
    
    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = Path(__file__).parent.parent / results_dir
    
    if not results_dir.exists():
        logger.error(f"Results directory not found: {results_dir}")
        return 1
    
    train_dir = results_dir / "train" / f"iter{args.iteration}"
    if not train_dir.exists():
        logger.error(f"Train directory not found: {train_dir}")
        return 1
    
    # Output paths (matching real pipeline)
    artifacts_dir = results_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    result_file = Path(args.output) if args.output else \
                  artifacts_dir / f"agent_voter_result_iter{args.iteration}.json"
    conv_file = artifacts_dir / f"agent_voter_conversation_iter{args.iteration}.json"
    
    # Show ground truth
    best_candidate, best_score = get_ground_truth(train_dir)
    candidates = sorted([d.name for d in train_dir.glob("sample*")], 
                       key=lambda x: int(x.replace("sample", "")))
    
    print(f"\n{'='*60}")
    print("CLAUDE AGENT ELITE VOTER TEST")
    print(f"{'='*60}")
    print(f"Results dir: {results_dir}")
    print(f"Train dir: {train_dir}")
    print(f"Iteration: {args.iteration}")
    print(f"Candidates: {len(candidates)}")
    print(f"Model: {model}")
    print(f"Result file: {result_file}")
    mode_str = "REAL AgentVoter" if args.use_real_voter else "STANDALONE"
    print(f"Mode: {mode_str}")
    
    if best_candidate:
        best_idx = candidates.index(best_candidate)
        print(f"\nGround truth: {best_candidate} (index {best_idx})")
        print(f"  success_rate={best_score[0]:.3f}, final_distance={-best_score[1]:.3f}m")
    print(f"{'='*60}\n")
    
    # Run agent
    try:
        if args.use_real_voter:
            # Use actual AgentVoter class
            from omegaconf import OmegaConf
            from quadro_llm.core.agent_voter import AgentVoter
            
            # Convert dict config to OmegaConf (like Hydra does)
            agent_config = OmegaConf.create(config)
            voter = AgentVoter(agent_config)
            vote_result = voter.vote(train_dir, results_dir, args.iteration)
            
            result = {
                "selected_index": vote_result.selected_index,
                "selected_identifier": vote_result.selected_identifier,
                "reasoning": vote_result.reasoning,
                "analysis_summary": vote_result.analysis_summary,
            }
        else:
            # Standalone implementation (same logic as AgentVoter)
            result = asyncio.run(run_agent_voter(
                train_dir, results_dir, result_file, conv_file, model
            ))
        
        print("\n" + "="*60)
        print("AGENT SELECTION RESULT")
        print("="*60)
        print(json.dumps(result, indent=2))
        
        if best_candidate:
            selected_id = result.get("selected_identifier", "")
            if selected_id == best_candidate:
                print("\n✅ CORRECT: Agent selected the best candidate!")
            else:
                print(f"\n⚠️  Agent selected: {selected_id}, ground truth was: {best_candidate}")
        
        return 0
            
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Agent voter failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Check if result file was created despite error
        if result_file.exists():
            print(f"\n⚠️  Error occurred but result file exists!")
            with open(result_file) as f:
                result = json.load(f)
            print(json.dumps(result, indent=2))
            print("\nThis confirms the bug: Agent succeeded but SDK threw error during cleanup.")
        else:
            print(f"\n❌ Result file not created: {result_file}")
        
        return 1


if __name__ == "__main__":
    exit(main())

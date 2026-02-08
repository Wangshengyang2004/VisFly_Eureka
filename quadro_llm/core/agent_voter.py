"""Agent-based voter using Claude Agent SDK for selecting best reward function."""

import os
import asyncio
import json
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from omegaconf import DictConfig

from ..utils.tensorboard_utils import find_tensorboard_logdir, load_tensorboard_logs


@dataclass
class AgentVoterResult:
    """Result from agent voter."""
    selected_index: int
    selected_identifier: str
    reasoning: str
    analysis_summary: str
    candidate_count: int
    conversation: Optional[List[Dict]] = None
    code_level_feedback: Optional["CodeLevelFeedback"] = None


class AgentVoter:
    """Claude Agent SDK-based voter for selecting elite reward functions.
    
    Uses autonomous agent to explore training data, analyze trajectories,
    and make informed selection decisions.
    """
    
    def __init__(self, config: DictConfig, prompt_config: Optional[DictConfig] = None):
        self.config = config
        self.logger = logging.getLogger(__name__)
        # Number of alternative_directions to request (from prompt config); 0 = off
        n = 3
        if prompt_config is not None:
            n_val = prompt_config.get("num_alternative_directions", 3)
            if n_val is not None:
                n = int(n_val)
        self.num_alternative_directions = n
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

    def _collect_tensorboard_metrics(self, train_dir: Path) -> Dict[str, Dict[str, List[float]]]:
        """Pre-parse TensorBoard logs for each sample using project's tensorboard package (no TensorFlow)."""
        metrics_by_sample: Dict[str, Dict[str, List[float]]] = {}
        for sample_dir in sorted(train_dir.glob("sample*"), key=lambda p: int(p.name.replace("sample", ""))):
            logdir = find_tensorboard_logdir(str(sample_dir))
            if logdir:
                logs = load_tensorboard_logs(logdir)
                if logs:
                    metrics_by_sample[sample_dir.name] = logs
                    self.logger.debug(f"Loaded TensorBoard metrics for {sample_dir.name}: {list(logs.keys())}")
            else:
                self.logger.debug(f"No TensorBoard logdir found under {sample_dir}")
        return metrics_by_sample

    def _build_history_file(self, output_dir: Path, iteration: int, history_window: int) -> Optional[Path]:
        """Concat previous N iterations' voter traces into a single markdown file.

        Returns path to the file, or None if no history exists.
        """
        if history_window <= 0 or iteration <= 0:
            return None

        artifacts_dir = output_dir / "artifacts"
        start_iter = max(0, iteration - history_window)

        sections = []
        for prev_iter in range(start_iter, iteration):
            result_file = artifacts_dir / f"agent_voter_result_iter{prev_iter}.json"
            conv_file = artifacts_dir / f"agent_voter_conversation_iter{prev_iter}.json"

            if not result_file.exists():
                continue

            sections.append(f"# Iteration {prev_iter}\n")
            sections.append(f"## Vote Result\n```json\n{result_file.read_text()}\n```\n")

            if conv_file.exists():
                sections.append(f"## Agent Conversation Trace\n```json\n{conv_file.read_text()}\n```\n")

        if not sections:
            return None

        history_file = output_dir / "voter_history.md"
        history_file.write_text("\n".join(sections))
        self.logger.info(f"Built voter history file with {len(sections)} sections from iter {start_iter}-{iteration - 1}")
        return history_file

    def vote(self, train_dir: Path, output_dir: Path, iteration: int, task_description: str,
             history_window: int = 0) -> AgentVoterResult:
        """Run agent to analyze candidates and select the best one.

        Args:
            train_dir: Path to training results (e.g., results/xxx/train/iter0)
            output_dir: Path to save artifacts (e.g., results/xxx)
            iteration: Current iteration number
            task_description: Description of the task to evaluate against
            history_window: Number of previous iterations' voter traces to include (0 = disabled)

        Returns:
            AgentVoterResult with selection and reasoning
        """
        return asyncio.run(self._vote_async(train_dir, output_dir, iteration, task_description, history_window))
    
    async def _vote_async(self, train_dir: Path, output_dir: Path, iteration: int, task_description: str, history_window: int = 0) -> AgentVoterResult:
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
        tb_metrics_file = artifacts_dir / f"tensorboard_metrics_iter{iteration}.json"

        # Count candidates
        candidates = sorted([d.name for d in train_dir.glob("sample*")],
                          key=lambda x: int(x.replace("sample", "")))

        # Pre-parse TensorBoard logs so the agent gets training curves without parsing tfevents (no TensorFlow needed)
        tb_metrics = self._collect_tensorboard_metrics(train_dir)
        if tb_metrics:
            with open(tb_metrics_file, "w") as f:
                json.dump(tb_metrics, f, indent=2)
            self.logger.info(f"Wrote pre-parsed TensorBoard metrics for {len(tb_metrics)} samples to {tb_metrics_file}")
        else:
            self.logger.debug("No TensorBoard metrics found; agent will rely on result.json and trajectories only")

        tb_instruction = (
            f"- Pre-parsed training curves: {tb_metrics_file} — use this for convergence and reward-curve comparison. "
            f"Do not parse .tfevents yourself and do not install or use TensorFlow; the JSON is sufficient.\n"
            if tb_metrics
            else "- Training curves: not available for this run; base your analysis on result.json and trajectories only.\n"
        )

        # Build cross-iteration history file for context continuity
        history_file = self._build_history_file(output_dir, iteration, history_window)
        history_instruction = ""
        if history_file:
            history_instruction = (
                f"FIRST read {history_file.name} in your working directory — "
                f"it contains the analysis and decisions from the previous {min(iteration, history_window)} iteration(s). "
                f"Use it to maintain consistency and reference prior component evaluations.\n"
            )

        self.logger.info(f"Agent voting on {len(candidates)} candidates from {train_dir}")

        n_dir = self.num_alternative_directions
        directions_instruction = ""
        directions_json_block = ""
        if n_dir >= 1:
            directions_instruction = (
                f"\nIMPORTANT: In your code_level_feedback, provide exactly {n_dir} alternative_directions for the next iteration.\n"
                "Each direction should explore a DIFFERENT approach to improving the reward function.\n"
                "This ensures diversity in the next generation of samples.\n"
            )
            direction_template = """            {{
                "direction_id": <id>,
                "focus": "<e.g., 'aggressive_terminal_shaping'>",
                "description": "<what to try>",
                "rationale": "<why this might work>",
                "suggested_changes": ["<specific change 1>", "<specific change 2>"]
            }}"""
            direction_entries = [direction_template.replace("<id>", str(i + 1)) for i in range(n_dir)]
            directions_json_block = ",\n        // Provide exactly {} different improvement directions for diversity\n        \"alternative_directions\": [\n".format(n_dir) + ",\n".join(direction_entries) + "\n        ]"
        else:
            directions_json_block = ""

        directions_json_inline = (
            ",\n" + directions_json_block if directions_json_block else ""
        )
        prompt = f"""Select the best reward function candidate for the task: {task_description}.

Perform a deep analysis for this task. Explore {train_dir} and analyze each sample using:
- result.json (metrics)
- trajectories/*.npz (trajectory data)
{tb_instruction}
{history_instruction}{directions_instruction}
Write your final decision to {result_file} as JSON with this structure:
{{
    "selected_index": <int>,
    "selected_identifier": "<sampleN>",
    "reasoning": "<why selected>",
    "analysis_summary": "<overview>",

    // Code-level feedback for the NEXT iteration - use simplified structure
    "code_level_feedback": {{
        // List problematic components as strings, e.g., "distance_penalty: -2.0 is too aggressive"
        "problematic_components": [
            "<component_name>: <description of issue with current value>"
        ],
        // List component names that should be added
        "missing_components": [
            "<component name that would improve performance>"
        ],
        // List successful patterns as strings, e.g., "velocity_penalty: -0.5 works well for stability"
        "successful_patterns": [
            "<component_name>: <description with value and effect>"
        ],
        // Training observations
        "training_insights": [
            "<observation about training dynamics>"
        ],
        // Convergence description or null
        "convergence_issues": "<description like 'oscillating at target' or null>"
{directions_json_inline}
    }}
}}

IMPORTANT: All values must be valid JSON. Use strings for descriptions, arrays for lists. Keep the structure flat and simple.

Focus on ACTIONABLE feedback that tells the LLM exactly what to change in the next iteration.
"""
        if n_dir >= 1:
            prompt += "Provide exactly {} alternative_directions to maintain diversity while avoiding low-quality code.\n".format(n_dir)

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
        
        # Read result written by the agent with error handling
        try:
            with open(result_file) as f:
                result_data = json.load(f)
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON parse error in {result_file}: {e}")
            # Create a minimal valid result with defaults
            result_data = {
                "selected_index": 0,
                "selected_identifier": candidates[0] if candidates else "sample0",
                "reasoning": f"Failed to parse agent result: {e}. Using fallback selection.",
                "analysis_summary": f"JSON parse error: {e}",
                "code_level_feedback": None,
            }

        candidate_count = len(candidates)
        agent_summary = result_data["analysis_summary"].strip()
        analysis_summary = f"Analyzed {candidate_count} candidates. {agent_summary}" if agent_summary else f"Analyzed {candidate_count} candidates."

        # Parse code_level_feedback if present
        code_level_feedback = None
        if "code_level_feedback" in result_data and result_data["code_level_feedback"]:
            from .models import (
                CodeLevelFeedback, ComponentIssue, SuccessfulPattern, AlternativeDirection
            )
            clf_data = result_data["code_level_feedback"]

            # Parse problematic_components - handle both old object format and new string format
            problematic_components = []
            for pc in clf_data.get("problematic_components", []):
                if not pc:
                    continue
                if isinstance(pc, str):
                    # New format: string like "component_name: description"
                    problematic_components.append(ComponentIssue(
                        component=pc.split(":")[0].strip() if ":" in pc else pc,
                        current_value=0.0,
                        issue=pc,
                        suggested_value=None,
                        suggested_action=None,
                    ))
                elif isinstance(pc, dict) and "component" in pc:
                    # Old format: object with fields
                    problematic_components.append(ComponentIssue(
                        component=pc["component"],
                        current_value=pc.get("current_value", 0.0),
                        issue=pc.get("issue", ""),
                        suggested_value=pc.get("suggested_value"),
                        suggested_action=pc.get("suggested_action"),
                    ))

            # Parse successful_patterns - handle both old object format and new string format
            successful_patterns = []
            for sp in clf_data.get("successful_patterns", []):
                if not sp:
                    continue
                if isinstance(sp, str):
                    # New format: string like "component_name: description"
                    successful_patterns.append(SuccessfulPattern(
                        component=sp.split(":")[0].strip() if ":" in sp else sp,
                        value=0.0,
                        effect=sp,
                    ))
                elif isinstance(sp, dict) and "component" in sp:
                    # Old format: object with fields
                    successful_patterns.append(SuccessfulPattern(
                        component=sp["component"],
                        value=sp.get("value", 0.0),
                        effect=sp.get("effect", ""),
                    ))

            # Parse alternative_directions with validation
            alternative_directions = []
            for ad in clf_data.get("alternative_directions", []):
                if not ad or not isinstance(ad, dict):
                    continue
                if "direction_id" not in ad or "focus" not in ad:
                    continue
                alternative_directions.append(AlternativeDirection(
                    direction_id=ad.get("direction_id", 0),
                    focus=ad.get("focus", ""),
                    description=ad.get("description", ""),
                    rationale=ad.get("rationale", ""),
                    suggested_changes=ad.get("suggested_changes", []),
                ))

            code_level_feedback = CodeLevelFeedback(
                problematic_components=problematic_components,
                missing_components=clf_data.get("missing_components", []),
                successful_patterns=successful_patterns,
                training_insights=clf_data.get("training_insights", []),
                convergence_issues=clf_data.get("convergence_issues"),
                alternative_directions=alternative_directions,
            )
            self.logger.info(
                f"Parsed code_level_feedback: {len(problematic_components)} issues, "
                f"{len(successful_patterns)} patterns, "
                f"{len(alternative_directions)} alternative directions"
            )

        # Persist with candidate_count and normalized summary for verification
        result_data["candidate_count"] = candidate_count
        result_data["analysis_summary"] = analysis_summary
        with open(result_file, "w") as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)

        return AgentVoterResult(
            selected_index=result_data["selected_index"],
            selected_identifier=result_data["selected_identifier"],
            reasoning=result_data["reasoning"],
            analysis_summary=analysis_summary,
            candidate_count=candidate_count,
            conversation=conversation,
            code_level_feedback=code_level_feedback,
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

"""Elite voter for selecting best reward function using LLM analysis."""

import json
import logging
import numpy as np
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass

import torch as th
from VisFly.utils.maths import Quaternion

from .models import RewardFunctionResult
from ..llm.llm_engine import LLMEngine


@dataclass
class EliteVoterResult:
    """Result from elite voter LLM agent"""
    selected_index: int
    reasoning: str
    confidence: Optional[float] = None


class EliteVoter:
    """LLM agent for voting/selecting elite reward functions"""
    
    def __init__(self, llm_engine: LLMEngine):
        self.llm_engine = llm_engine
        self.logger = logging.getLogger(__name__)
    
    def format_trajectory_data(self, trajectory_path: str, is_flip_task: bool = False) -> str:
        """Format trajectory npz data for LLM prompt"""
        data = np.load(trajectory_path)
        
        positions = data["positions"]
        velocities = data["velocities"]
        orientations = data["orientations"]
        angular_velocities = data["angular_velocities"]
        target = data.get("target")
        
        if positions.ndim == 3 and positions.shape[1] > 1:
            positions = positions.mean(axis=1)
            velocities = velocities.mean(axis=1)
            orientations = orientations.mean(axis=1)
            angular_velocities = angular_velocities.mean(axis=1)
        elif positions.ndim == 3:
            positions = positions.squeeze(1)
            velocities = velocities.squeeze(1)
            orientations = orientations.squeeze(1)
            angular_velocities = angular_velocities.squeeze(1)
        
        timesteps = positions.shape[0]
        lines = [f"Trajectory data (ALL {timesteps} timesteps, no sampling):"]
        
        if is_flip_task:
            lines.append("\n[NOTE: This is a FLIP task - orientation changes and trajectory smoothness are CRITICAL for evaluation]")
        
        lines.append(f"\nPosition (x, y, z) for ALL {timesteps} steps:")
        for i in range(timesteps):
            pos = positions[i]
            lines.append(f"  Step {i}: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
        
        lines.append(f"\nOrientation (quaternion [w, x, y, z]) for ALL {timesteps} steps - {'CRITICAL for flip evaluation' if is_flip_task else 'orientation data'}:")
        for i in range(timesteps):
            ori = orientations[i]
            lines.append(f"  Step {i}: [{ori[0]:.3f}, {ori[1]:.3f}, {ori[2]:.3f}, {ori[3]:.3f}]")
        
        if is_flip_task:
            lines.append(f"\nOrientation (Euler angles [Roll, Pitch, Yaw] in degrees) for ALL {timesteps} steps - CRITICAL for flip task:")
            for i in range(timesteps):
                q_arr = orientations[i]
                q = Quaternion(
                    w=th.tensor(q_arr[0]),
                    x=th.tensor(q_arr[1]),
                    y=th.tensor(q_arr[2]),
                    z=th.tensor(q_arr[3])
                )
                euler = q.toEuler(order="zyx")
                euler_deg = np.degrees(euler.detach().cpu().numpy())
                lines.append(f"  Step {i}: Roll={euler_deg[0]:.1f}°, Pitch={euler_deg[1]:.1f}°, Yaw={euler_deg[2]:.1f}°")

            q_start = Quaternion(
                w=th.tensor(orientations[0][0]),
                x=th.tensor(orientations[0][1]),
                y=th.tensor(orientations[0][2]),
                z=th.tensor(orientations[0][3])
            )
            q_end = Quaternion(
                w=th.tensor(orientations[-1][0]),
                x=th.tensor(orientations[-1][1]),
                y=th.tensor(orientations[-1][2]),
                z=th.tensor(orientations[-1][3])
            )
            q_diff = q_start.inverse() * q_end
            rotation_angle = 2 * th.acos(th.clamp(th.abs(q_diff.w), 0, 1))
            rotation_angle_deg = np.degrees(rotation_angle.detach().cpu().numpy())
            lines.append(f"\nTotal orientation change: {rotation_angle_deg:.1f}° (360° = full flip)")
        
        lines.append(f"\nVelocity (vx, vy, vz) for ALL {timesteps} steps:")
        for i in range(timesteps):
            vel = velocities[i]
            lines.append(f"  Step {i}: [{vel[0]:.3f}, {vel[1]:.3f}, {vel[2]:.3f}]")
        
        lines.append(f"\nAngular velocity (ωx, ωy, ωz in rad/s) for ALL {timesteps} steps{' - CRITICAL for flip task' if is_flip_task else ''}:")
        for i in range(timesteps):
            angvel = angular_velocities[i]
            lines.append(f"  Step {i}: [{angvel[0]:.3f}, {angvel[1]:.3f}, {angvel[2]:.3f}]")
        
        if target is not None:
            if target.ndim == 1:
                target_str = f"[{target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f}]"
            else:
                target_str = f"[{target[0,0]:.3f}, {target[0,1]:.3f}, {target[0,2]:.3f}]"
            lines.append(f"\nTarget position: {target_str}")
        
        return "\n".join(lines)
    
    def format_candidate_summary(self, result: RewardFunctionResult, index: int, is_flip_task: bool = False) -> str:
        """Format a single candidate's summary for LLM prompt"""
        lines = [f"=== Candidate {index} (ID: {result.identifier}) ==="]
        
        if result.evaluation_summary:
            eval_sum = result.evaluation_summary
            lines.append(f"Evaluation Results:")
            lines.append(f"  Success Rate: {eval_sum.get('success_rate', result.success_rate):.3f}")
            lines.append(f"  Success Count: {eval_sum.get('success_count', 0)} / {eval_sum.get('actual_evaluation_episodes', 'N/A')} episodes")
            if 'mean_episode_length' in eval_sum:
                lines.append(f"  Episode Length: {eval_sum.get('mean_episode_length', 0):.1f}")
            if 'mean_final_distance' in eval_sum and not np.isnan(eval_sum.get('mean_final_distance', np.nan)):
                lines.append(f"  Final Distance to Target: {eval_sum.get('mean_final_distance', 0):.3f}m")
            if 'collision_count' in eval_sum:
                lines.append(f"  Collisions: {eval_sum.get('collision_count', 0)}")
        
        if result.episode_statistics and len(result.episode_statistics) > 0:
            num_eval_episodes = len(result.episode_statistics)
            num_traj_files = sum(1 for ep in result.episode_statistics if ep.get("trajectory_path") and Path(ep.get("trajectory_path")).exists())

            # Use first episode's trajectory (all candidates' trajectories will be shown together to LLM)
            traj_path = result.episode_statistics[0].get("trajectory_path")
            if traj_path and Path(traj_path).exists():
                task_is_flip = is_flip_task or "flip" in result.identifier.lower() or any(
                    "flip" in str(stat.get("env_name", "")).lower()
                    for stat in result.episode_statistics
                )
                if num_eval_episodes > 1:
                    lines.append(f"\nTrajectory Data (first episode of {num_eval_episodes} eval episodes, {num_traj_files} files total):")
                else:
                    lines.append(f"\nTrajectory Data ({num_eval_episodes} eval episode):")
                traj_str = self.format_trajectory_data(traj_path, is_flip_task=task_is_flip)
                lines.append(traj_str)
            else:
                self.logger.warning(
                    f"No valid trajectory file for candidate {result.identifier} "
                    f"(eval_episodes: {num_eval_episodes}, traj_files: {num_traj_files})"
                )
                lines.append(f"\nTrajectory Data: Not available ({num_eval_episodes} eval episodes, {num_traj_files} files)")
        else:
            lines.append(f"\nTrajectory Data: No episode statistics available")
        
        lines.append(f"\nTraining:")
        lines.append(f"  Training Successful: {result.training_successful}")
        lines.append(f"  Training Time: {result.training_time:.1f}s")
        
        return "\n".join(lines)
    
    def vote(self, results: List[RewardFunctionResult]) -> EliteVoterResult:
        """Use LLM to vote/select the best reward function from candidates."""
        if not results:
            raise ValueError("Cannot vote on empty results list")
        
        if len(results) == 1:
            return EliteVoterResult(selected_index=0, reasoning="Only one candidate available", confidence=1.0)
        
        successful_results = [r for r in results if r.training_successful]
        if not successful_results:
            return EliteVoterResult(selected_index=0, reasoning="All candidates failed training", confidence=0.0)
        
        is_flip_task = any("flip" in r.identifier.lower() for r in results)
        
        system_prompt = """You are an elite voter agent selecting the best reward function from multiple candidates.

Analyze the evaluation results and trajectory data to select the best candidate.

Focus on:
1. Evaluation success rate (most important)
2. Trajectory quality from the raw trajectory and orientation data
3. Task completion (e.g., for flip tasks: does it complete a full flip, is it smooth vs trembling)

Do NOT consider reward values as they are scaled differently by different reward functions.

Respond with ONLY a JSON object:
{
    "selected_index": <int>,
    "reasoning": "<Focus on why the selected candidate is elite: (1) describe its trajectory and orientation characteristics that make it superior, (2) highlight its evaluation metrics, (3) explain why it outperforms others. Keep it concise and focused on the selected elite candidate>",
    "confidence": <float between 0.0 and 1.0>
}

The 'reasoning' field should:
- Focus primarily on the selected elite candidate
- Explain why it is the best choice
- Highlight key strengths (trajectory quality, orientation control, task completion)
- Be concise and actionable"""
        
        user_prompt_parts = []
        if is_flip_task:
            user_prompt_parts.append(
                "Analyzing FLIP task candidates. For each candidate, examine:\n"
                "- Evaluation success rate\n"
                "- Trajectory and orientation data: does it complete a full flip (360° rotation)? Is it smooth or trembling?\n"
                "- Position stability during flip\n\n"
            )
        
        user_prompt_parts.append("Analyze the following candidates:\n")
        user_prompt_parts.append("=" * 60)
        user_prompt_parts.append("")
        
        successful_indices = []
        for i, result in enumerate(results):
            if result.training_successful:
                successful_indices.append(i)
                candidate_idx = len(successful_indices) - 1
                user_prompt_parts.append(self.format_candidate_summary(result, candidate_idx, is_flip_task=is_flip_task))
                user_prompt_parts.append("")
        
        user_prompt = "\n".join(user_prompt_parts)
        user_prompt += "\n\nBased on your analysis of trajectory and orientation for each candidate, "
        user_prompt += "select the best candidate and provide a JSON object with 'selected_index', 'reasoning', and 'confidence' fields. "
        user_prompt += "The 'selected_index' should be the candidate number shown above (0-based). "
        user_prompt += "The 'reasoning' should focus on why the selected candidate is elite - describe its superior trajectory/orientation characteristics and evaluation performance."
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Use _build_request_params to ensure all LLM config (including thinking) is applied
        request_params = self.llm_engine._build_request_params(
            messages=messages,
            max_tokens_override=None,  # No limit for voting
            timeout_override=None,
        )
        # Override temperature for voting (lower temperature for more deterministic selection)
        request_params["temperature"] = 0.3
        
        response = self.llm_engine.client.chat.completions.create(**request_params)
        
        content = response.choices[0].message.content.strip()
        
        if "```json" in content:
            json_start = content.find("```json") + 7
            json_end = content.find("```", json_start)
            content = content[json_start:json_end].strip()
        elif "```" in content:
            json_start = content.find("```") + 3
            json_end = content.find("```", json_start)
            content = content[json_start:json_end].strip()
        
        if not content.startswith("{"):
            start_idx = content.find("{")
            end_idx = content.rfind("}")
            if start_idx != -1 and end_idx != -1:
                content = content[start_idx:end_idx+1]

        try:
            vote_data = json.loads(content)
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse LLM response as JSON: {e}\nContent was: {content[:500]}")
            # Fallback to first successful candidate
            return EliteVoterResult(
                selected_index=successful_indices[0],
                reasoning=f"JSON parsing failed: {str(e)}",
                confidence=0.0
            )

        # Check for required field
        if "selected_index" not in vote_data:
            self.logger.error(f"LLM response missing 'selected_index' field. Keys: {list(vote_data.keys())}")
            return EliteVoterResult(
                selected_index=successful_indices[0],
                reasoning="LLM response missing required field 'selected_index'",
                confidence=0.0
            )

        selected_candidate_idx = vote_data["selected_index"]
        if selected_candidate_idx < 0 or selected_candidate_idx >= len(successful_indices):
            self.logger.warning(f"LLM selected invalid candidate index {selected_candidate_idx}, using first successful")
            selected_idx = successful_indices[0]
        else:
            selected_idx = successful_indices[selected_candidate_idx]
        
        return EliteVoterResult(
            selected_index=selected_idx,
            reasoning=vote_data.get("reasoning", "No reasoning provided"),
            confidence=vote_data.get("confidence", 0.5)
        )


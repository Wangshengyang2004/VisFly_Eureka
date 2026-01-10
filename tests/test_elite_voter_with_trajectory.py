"""
Test script for elite_voter LLM agent with trajectory data.

This script tests using an LLM agent to vote/select the best reward function
based on evaluation results including raw trajectory and orientation data.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import sys

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from quadro_llm.core.models import RewardFunctionResult
from quadro_llm.llm.llm_engine import LLMEngine


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
        self.logger = __import__('logging').getLogger(__name__)
    
    def format_trajectory_data(self, trajectory_path: str, is_flip_task: bool = False) -> str:
        """Format trajectory npz data for LLM prompt"""
        try:
            data = np.load(trajectory_path)
            
            # Extract arrays
            positions = data["positions"]  # Shape: (timesteps, num_agents, 3)
            velocities = data["velocities"]  # Shape: (timesteps, num_agents, 3)
            orientations = data["orientations"]  # Shape: (timesteps, num_agents, 4) [w, x, y, z]
            angular_velocities = data["angular_velocities"]  # Shape: (timesteps, num_agents, 3)
            target = data.get("target")  # Optional: (num_agents, 3) or (3,)
            
            # Take mean across agents if multiple agents
            if positions.ndim == 3 and positions.shape[1] > 1:
                positions = positions.mean(axis=1)  # (timesteps, 3)
                velocities = velocities.mean(axis=1)
                orientations = orientations.mean(axis=1)
                angular_velocities = angular_velocities.mean(axis=1)
            elif positions.ndim == 3:
                positions = positions.squeeze(1)  # (timesteps, 3)
                velocities = velocities.squeeze(1)
                orientations = orientations.squeeze(1)
                angular_velocities = angular_velocities.squeeze(1)
            
            # Format as readable string - NO SAMPLING, ALL DATA
            timesteps = positions.shape[0]
            lines = [f"Trajectory data (ALL {timesteps} timesteps, no sampling):"]
            
            if is_flip_task:
                lines.append("\n[NOTE: This is a FLIP task - orientation changes and trajectory smoothness are CRITICAL for evaluation]")
            
            # Position trajectory - ALL steps
            lines.append(f"\nPosition (x, y, z) for ALL {timesteps} steps:")
            for i in range(timesteps):
                pos = positions[i]
                lines.append(f"  Step {i}: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
            
            # Orientation - ALL steps, CRITICAL for flip tasks
            lines.append(f"\nOrientation (quaternion [w, x, y, z]) for ALL {timesteps} steps - {'CRITICAL for flip evaluation' if is_flip_task else 'orientation data'}:")
            for i in range(timesteps):
                ori = orientations[i]
                lines.append(f"  Step {i}: [{ori[0]:.3f}, {ori[1]:.3f}, {ori[2]:.3f}, {ori[3]:.3f}]")
            
            # Convert quaternion to Euler angles for better interpretability (especially for flip) - ALL steps
            if is_flip_task:
                try:
                    import torch as th
                    from VisFly.utils.maths import Quaternion
                    
                    lines.append(f"\nOrientation (Euler angles [Roll, Pitch, Yaw] in degrees) for ALL {timesteps} steps - CRITICAL for flip task:")
                    for i in range(timesteps):
                        q_arr = orientations[i]
                        q = Quaternion(
                            w=th.tensor(q_arr[0]),
                            x=th.tensor(q_arr[1]),
                            y=th.tensor(q_arr[2]),
                            z=th.tensor(q_arr[3])
                        )
                        euler = q.toEuler(order="zyx")  # Returns [roll, pitch, yaw] in radians
                        euler_deg = np.degrees(euler.detach().cpu().numpy())
                        lines.append(f"  Step {i}: Roll={euler_deg[0]:.1f}°, Pitch={euler_deg[1]:.1f}°, Yaw={euler_deg[2]:.1f}°")
                    
                    # Compute orientation change summary
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
                    # Compute rotation angle between start and end
                    q_diff = q_start.inverse() * q_end
                    rotation_angle = 2 * th.acos(th.clamp(th.abs(q_diff.w), 0, 1))
                    rotation_angle_deg = np.degrees(rotation_angle.detach().cpu().numpy())
                    lines.append(f"\nTotal orientation change: {rotation_angle_deg:.1f}° (360° = full flip)")
                except Exception as e:
                    self.logger.debug(f"Could not compute Euler angles: {e}")
            
            # Velocity - ALL steps
            lines.append(f"\nVelocity (vx, vy, vz) for ALL {timesteps} steps:")
            for i in range(timesteps):
                vel = velocities[i]
                lines.append(f"  Step {i}: [{vel[0]:.3f}, {vel[1]:.3f}, {vel[2]:.3f}]")
            
            # Angular velocity - ALL steps, important for flip
            lines.append(f"\nAngular velocity (ωx, ωy, ωz in rad/s) for ALL {timesteps} steps{' - CRITICAL for flip task' if is_flip_task else ''}:")
            for i in range(timesteps):
                angvel = angular_velocities[i]
                lines.append(f"  Step {i}: [{angvel[0]:.3f}, {angvel[1]:.3f}, {angvel[2]:.3f}]")
            
            # No complex diagnostic metrics - just provide raw data for LLM to analyze
            
            if target is not None:
                if target.ndim == 1:
                    target_str = f"[{target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f}]"
                else:
                    target_str = f"[{target[0,0]:.3f}, {target[0,1]:.3f}, {target[0,2]:.3f}]"
                lines.append(f"\nTarget position: {target_str}")
            
            return "\n".join(lines)
        except Exception as e:
            self.logger.warning(f"Failed to load trajectory from {trajectory_path}: {e}")
            return f"[Trajectory data unavailable: {e}]"
    
    def format_candidate_summary(self, result: RewardFunctionResult, index: int, is_flip_task: bool = False) -> str:
        """Format a single candidate's summary for LLM prompt"""
        lines = [f"=== Candidate {index} (ID: {result.identifier}) ==="]
        
        # Evaluation summary (most important)
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
        
        # Trajectory data (if available) - most important for analysis
        if result.episode_statistics and len(result.episode_statistics) > 0:
            # Get trajectory path from first episode
            traj_path = result.episode_statistics[0].get("trajectory_path")
            if traj_path and Path(traj_path).exists():
                # Detect if this is a flip task
                task_is_flip = is_flip_task or "flip" in result.identifier.lower() or any(
                    "flip" in str(stat.get("env_name", "")).lower() 
                    for stat in result.episode_statistics
                )
                lines.append(f"\nTrajectory Data:")
                traj_str = self.format_trajectory_data(traj_path, is_flip_task=task_is_flip)
                lines.append(traj_str)
        
        # Training stats (simple and concise)
        lines.append(f"\nTraining:")
        lines.append(f"  Training Successful: {result.training_successful}")
        lines.append(f"  Training Time: {result.training_time:.1f}s")
        
        return "\n".join(lines)
    
    def vote(self, results: List[RewardFunctionResult]) -> EliteVoterResult:
        """
        Use LLM to vote/select the best reward function from candidates.
        
        Args:
            results: List of RewardFunctionResult objects from one iteration
            
        Returns:
            EliteVoterResult with selected index and reasoning
        """
        if not results:
            raise ValueError("Cannot vote on empty results list")
        
        if len(results) == 1:
            return EliteVoterResult(
                selected_index=0,
                reasoning="Only one candidate available",
                confidence=1.0
            )
        
        # Filter to successful results only
        successful_results = [r for r in results if r.training_successful]
        if not successful_results:
            # If all failed, return first one with explanation
            return EliteVoterResult(
                selected_index=0,
                reasoning="All candidates failed training",
                confidence=0.0
            )
        
        # Detect if this is a flip task
        is_flip_task = any("flip" in r.identifier.lower() for r in results)
        
        # Build prompt - simple and concise
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
    "reasoning": "<brief analysis of each candidate's evaluation results and trajectory, then selection rationale>",
    "confidence": <float between 0.0 and 1.0>
}"""
        
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
        
        # Map successful results to original indices
        successful_indices = []
        for i, result in enumerate(results):
            if result.training_successful:
                successful_indices.append(i)
                user_prompt_parts.append(self.format_candidate_summary(result, i, is_flip_task=is_flip_task))
                user_prompt_parts.append("")  # Empty line between candidates
        
        user_prompt = "\n".join(user_prompt_parts)
        user_prompt += "\n\nBased on your analysis of trajectory and orientation for each candidate, "
        user_prompt += "provide your selection as a JSON object with 'selected_index', 'reasoning', and 'confidence' fields. "
        user_prompt += "The 'reasoning' should include trajectory/orientation analysis and comparison."
        
        # Call LLM
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            self.logger.info(f"Calling LLM API with model: {self.llm_engine.model}")
            
            response = self.llm_engine.client.chat.completions.create(
                model=self.llm_engine.model,
                messages=messages,
                temperature=0.3,  # Lower temperature for more deterministic voting
                # No max_tokens limit - let the model generate full response with all trajectory data
            )
            
            if not response.choices or len(response.choices) == 0:
                raise ValueError("LLM response has no choices")
            
            choice = response.choices[0]
            content = choice.message.content
            if content is None:
                raise ValueError("LLM response content is None")
            
            content = content.strip()
            self.logger.debug(f"LLM response received: {content[:200]}...")
            
            # Parse JSON response
            # Try to extract JSON from markdown code blocks if present
            if "```json" in content:
                json_start = content.find("```json") + 7
                json_end = content.find("```", json_start)
                content = content[json_start:json_end].strip()
            elif "```" in content:
                json_start = content.find("```") + 3
                json_end = content.find("```", json_start)
                content = content[json_start:json_end].strip()
            
            vote_data = json.loads(content)
            
            # Validate and map index
            selected_idx = vote_data["selected_index"]
            if selected_idx not in successful_indices:
                self.logger.warning(f"LLM selected index {selected_idx} which is not in successful indices {successful_indices}. Using first successful.")
                selected_idx = successful_indices[0]
            
            return EliteVoterResult(
                selected_index=selected_idx,
                reasoning=vote_data.get("reasoning", "No reasoning provided"),
                confidence=vote_data.get("confidence", 0.5)
            )
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse LLM response as JSON: {e}")
            self.logger.error(f"Response content: {content}")
            # Fallback to first successful
            return EliteVoterResult(
                selected_index=successful_indices[0],
                reasoning=f"LLM response parsing failed, using first successful candidate. Error: {e}",
                confidence=0.0
            )
        except Exception as e:
            self.logger.error(f"Error calling LLM for voting: {e}")
            # Fallback to first successful
            return EliteVoterResult(
                selected_index=successful_indices[0],
                reasoning=f"LLM voting failed, using first successful candidate. Error: {e}",
                confidence=0.0
            )


def load_result_from_json(result_json_path: Path, reward_code_path: Optional[Path] = None) -> RewardFunctionResult:
    """Load RewardFunctionResult from result.json file"""
    import json
    from quadro_llm.utils.tensorboard_utils import find_tensorboard_logdir, load_tensorboard_logs
    
    with open(result_json_path, 'r') as f:
        result_data = json.load(f)
    
    # Load reward code if available
    reward_code = ""
    if reward_code_path and reward_code_path.exists():
        reward_code = reward_code_path.read_text()
    elif result_json_path.parent / "reward_function.py" in result_json_path.parent.iterdir():
        reward_code = (result_json_path.parent / "reward_function.py").read_text()
    else:
        reward_code = result_data.get('reward_code', '')
    
    # Find tensorboard logs
    output_dir = result_json_path.parent
    log_dir_path = find_tensorboard_logdir(str(output_dir))
    tensorboard_logs = None
    if log_dir_path:
        try:
            tensorboard_logs = load_tensorboard_logs(log_dir_path)
        except Exception:
            pass
    
    # Fix trajectory paths in episode_statistics (if they exist but are relative)
    episode_statistics = result_data.get('episode_statistics', [])
    if episode_statistics:
        trajectories_dir = output_dir / "trajectories"
        for ep_stat in episode_statistics:
            # If trajectory_path is missing, try to find it based on episode_index
            if 'trajectory_path' not in ep_stat or not Path(ep_stat.get('trajectory_path', '')).exists():
                episode_idx = ep_stat.get('episode_index', 0)
                traj_path = trajectories_dir / f"episode_{episode_idx:03d}.npz"
                if traj_path.exists():
                    ep_stat['trajectory_path'] = str(traj_path)
            elif 'trajectory_path' in ep_stat:
                traj_path = Path(ep_stat['trajectory_path'])
                # If path doesn't exist, try relative to output_dir
                if not traj_path.exists():
                    rel_path = trajectories_dir / traj_path.name
                    if rel_path.exists():
                        ep_stat['trajectory_path'] = str(rel_path)
    
    return RewardFunctionResult(
        reward_code=reward_code,
        identifier=result_data['identifier'],
        training_successful=result_data.get('success', False),
        success_rate=result_data.get('success_rate', 0.0),
        episode_length=result_data.get('episode_length', 0.0),
        training_time=result_data.get('training_time', 0.0),
        final_reward=result_data.get('final_reward', 0.0),
        convergence_step=result_data.get('convergence_step', 0),
        log_dir=log_dir_path,
        peak_memory_mb=result_data.get('peak_memory_mb', 0.0),
        tensorboard_logs=tensorboard_logs,
        evaluation_summary=result_data.get('aggregate_statistics') or result_data.get('evaluation_summary'),
        episode_statistics=episode_statistics,
        video_paths=result_data.get('video_paths', []),
    )


def create_mock_result(identifier: str, success_rate: float, episode_length: float, 
                      trajectory_path: Optional[str] = None) -> RewardFunctionResult:
    """Create a mock RewardFunctionResult for testing"""
    return RewardFunctionResult(
        reward_code=f"# Mock reward code for {identifier}",
        identifier=identifier,
        training_successful=True,
        success_rate=success_rate,
        episode_length=episode_length,
        training_time=100.0,
        final_reward=success_rate * 100.0,
        convergence_step=1000,
        tensorboard_logs={
            "ep_rew_mean": [10.0, 20.0, 30.0, 40.0, 50.0],
            "success_rate": [0.2, 0.4, 0.6, 0.8, success_rate],
        },
        evaluation_summary={
            "success_rate": success_rate,
            "mean_episode_length": episode_length,
            "std_episode_length": episode_length * 0.1,
            "mean_episode_reward": success_rate * 100.0,
            "actual_evaluation_episodes": 1,
            "success_count": 1 if success_rate > 0.5 else 0,
        },
        episode_statistics=[
            {
                "episode_index": 0,
                "steps": episode_length,
                "success": success_rate > 0.5,
                "trajectory_path": trajectory_path,
            }
        ] if trajectory_path else [],
    )


def test_elite_voter_basic():
    """Basic test of elite voter functionality"""
    print("=" * 60)
    print("Test: Basic Elite Voter")
    print("=" * 60)
    
    # Create mock LLM engine using aigc35 and gpt-5-nano
    from quadro_llm.bootstrap import load_llm_config
    from omegaconf import OmegaConf
    
    # Load config using Hydra compose
    from hydra import initialize_config_dir, compose
    from hydra.core.global_hydra import GlobalHydra
    
    # Clear any existing Hydra instance
    GlobalHydra.instance().clear()
    
    config_dir = project_root / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        # Compose config with gpt-5-nano override
        cfg = compose(config_name="config", overrides=["llm=gpt-5-nano"])
    
    llm_config = load_llm_config(cfg)
    print(f"Using LLM model: {llm_config.get('model', 'unknown')}")
    print(f"Using base_url: {llm_config.get('base_url', 'unknown')[:50]}...")
    
    llm_engine = LLMEngine(**llm_config)
    voter = EliteVoter(llm_engine)
    
    # Create mock results with different success rates
    results = [
        create_mock_result("candidate_0", 0.85, 120.0),
        create_mock_result("candidate_1", 0.70, 110.0),
        create_mock_result("candidate_2", 0.95, 115.0),  # Best success rate
        create_mock_result("candidate_3", 0.60, 100.0),
    ]
    
    print(f"\nVoting on {len(results)} candidates...")
    vote_result = voter.vote(results)
    
    print(f"\nVote Result:")
    print(f"  Selected Index: {vote_result.selected_index}")
    print(f"  Reasoning: {vote_result.reasoning}")
    print(f"  Confidence: {vote_result.confidence}")
    
    # Verify selection
    selected_result = results[vote_result.selected_index]
    print(f"\nSelected Candidate:")
    print(f"  ID: {selected_result.identifier}")
    print(f"  Success Rate: {selected_result.success_rate:.3f}")
    print(f"  Episode Length: {selected_result.episode_length:.1f}")
    
    return vote_result


def test_elite_voter_with_real_results(results_dir: Path):
    """Test elite voter with real evaluation results from a results directory"""
    print("\n" + "=" * 60)
    print("Test: Elite Voter with Real Evaluation Results")
    print("=" * 60)
    
    # Find result.json files (pattern: train/iter*/sample*/result.json)
    result_files = []
    
    # Try train/iter*/sample*/result.json pattern
    result_files.extend(list(results_dir.glob("train/iter*/sample*/result.json")))
    
    # Also try direct result.json in subdirectories
    if not result_files:
        result_files.extend(list(results_dir.glob("*/result.json")))
    
    if not result_files:
        print(f"No result.json files found in {results_dir}")
        print(f"Tried patterns:")
        print(f"  - train/iter*/sample*/result.json")
        print(f"  - */result.json")
        return None
    
    # Sort and limit to one iteration's samples (take first 5)
    result_files = sorted(result_files)[:5]
    print(f"Found {len(result_files)} result files")
    
    # Load LLM config
    from quadro_llm.bootstrap import load_llm_config
    from hydra import initialize_config_dir, compose
    from hydra.core.global_hydra import GlobalHydra
    
    # Clear any existing Hydra instance
    GlobalHydra.instance().clear()
    
    config_dir = project_root / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        # Compose config with gpt-5-nano override
        cfg = compose(config_name="config", overrides=["llm=gpt-5-nano"])
    
    llm_config = load_llm_config(cfg)
    print(f"Using LLM model: {llm_config.get('model', 'unknown')}")
    llm_engine = LLMEngine(**llm_config)
    voter = EliteVoter(llm_engine)
    
    # Load real results
    results = []
    for result_file in result_files:
        try:
            result = load_result_from_json(result_file)
            results.append(result)
            print(f"Loaded: {result.identifier} (success_rate={result.success_rate:.3f})")
        except Exception as e:
            print(f"Failed to load {result_file}: {e}")
            import traceback
            traceback.print_exc()
    
    if not results:
        print("No valid results loaded")
        return None
    
    print(f"\nVoting on {len(results)} candidates with real evaluation data...")
    vote_result = voter.vote(results)
    
    print(f"\nVote Result:")
    print(f"  Selected Index: {vote_result.selected_index}")
    print(f"  Reasoning: {vote_result.reasoning}")
    print(f"  Confidence: {vote_result.confidence}")
    
    selected_result = results[vote_result.selected_index]
    print(f"\nSelected Candidate:")
    print(f"  ID: {selected_result.identifier}")
    print(f"  Success Rate: {selected_result.success_rate:.3f}")
    print(f"  Episode Length: {selected_result.episode_length:.1f}")
    if selected_result.evaluation_summary:
        print(f"  Evaluation Episodes: {selected_result.evaluation_summary.get('actual_evaluation_episodes', 'N/A')}")
    
    return vote_result


def test_elite_voter_with_trajectory(trajectory_dir: Path):
    """Test elite voter with real trajectory data (deprecated, use test_elite_voter_with_real_results)"""
    print("\n" + "=" * 60)
    print("Test: Elite Voter with Trajectory Data")
    print("=" * 60)
    
    # Find trajectory npz files
    trajectory_files = list(trajectory_dir.glob("*.npz"))
    if not trajectory_files:
        print(f"No trajectory files found in {trajectory_dir}")
        return None
    
    print(f"Found {len(trajectory_files)} trajectory files")
    
    # Load LLM config
    from quadro_llm.bootstrap import load_llm_config
    from hydra import initialize_config_dir, compose
    from hydra.core.global_hydra import GlobalHydra
    
    # Clear any existing Hydra instance
    GlobalHydra.instance().clear()
    
    config_dir = project_root / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        # Compose config with gpt-5-nano override
        cfg = compose(config_name="config", overrides=["llm=gpt-5-nano"])
    
    llm_config = load_llm_config(cfg)
    print(f"Using LLM model: {llm_config.get('model', 'unknown')}")
    llm_engine = LLMEngine(**llm_config)
    voter = EliteVoter(llm_engine)
    
    # Create results with trajectory paths
    results = []
    for i, traj_file in enumerate(trajectory_files[:4]):  # Test with up to 4 files
        results.append(create_mock_result(
            f"candidate_{i}",
            success_rate=0.5 + i * 0.1,  # Varying success rates
            episode_length=100.0 + i * 10.0,
            trajectory_path=str(traj_file)
        ))
    
    print(f"\nVoting on {len(results)} candidates with trajectory data...")
    vote_result = voter.vote(results)
    
    print(f"\nVote Result:")
    print(f"  Selected Index: {vote_result.selected_index}")
    print(f"  Reasoning: {vote_result.reasoning}")
    print(f"  Confidence: {vote_result.confidence}")
    
    return vote_result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test elite voter with evaluation results")
    parser.add_argument(
        "--results-dir",
        type=Path,
        help="Results directory containing result.json files (e.g., results/2025-12-29_19-35-01/train/iter0)"
    )
    parser.add_argument(
        "--trajectory-dir",
        type=Path,
        help="Directory containing trajectory .npz files to test with (deprecated)"
    )
    args = parser.parse_args()
    
    # Run basic test
    try:
        test_elite_voter_basic()
    except Exception as e:
        print(f"Basic test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Run real results test if directory provided
    if args.results_dir and args.results_dir.exists():
        try:
            test_elite_voter_with_real_results(args.results_dir)
        except Exception as e:
            print(f"Real results test failed: {e}")
            import traceback
            traceback.print_exc()
    elif args.trajectory_dir and args.trajectory_dir.exists():
        try:
            test_elite_voter_with_trajectory(args.trajectory_dir)
        except Exception as e:
            print(f"Trajectory test failed: {e}")
            import traceback
            traceback.print_exc()


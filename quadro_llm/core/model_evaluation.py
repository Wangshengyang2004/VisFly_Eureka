"""
Evaluation utilities for trained models in VisFly environments.
"""

import torch
import logging
from typing import Tuple, Any, Optional
import time


class ModelEvaluator:
    """Evaluates trained models in VisFly environments"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
    
    def evaluate_trained_model(
        self, 
        model: Any, 
        env: Any, 
        num_episodes: int
    ) -> Tuple[float, float, float]:
        """
        Evaluate a trained model's performance.
        
        Args:
            model: Trained model with predict or get_action method
            env: Environment instance
            num_episodes: Number of evaluation episodes
            
        Returns:
            Tuple of (success_rate, avg_episode_length, avg_final_reward)
        """
        successes = 0
        episode_lengths = []
        final_rewards = []
        
        for episode in range(num_episodes):
            obs = env.reset()
            done = False
            step_count = 0
            episode_reward = 0
            
            while not done and step_count < env.max_episode_steps:
                if hasattr(model, 'predict'):
                    # Stable Baselines3 model
                    action, _ = model.predict(obs, deterministic=True)
                else:
                    # Custom BPTT model - need to implement predict method
                    action = model.get_action(obs)
                    
                obs, reward, done, info = env.step(action)
                episode_reward += reward.mean().item() if isinstance(reward, torch.Tensor) else reward
                step_count += 1
            
            # Check success condition
            if hasattr(env, 'get_success') and env.get_success().any():
                successes += 1
            elif episode_reward > 0:  # Fallback success criteria
                successes += 1
                
            episode_lengths.append(step_count)
            final_rewards.append(episode_reward)
        
        success_rate = successes / num_episodes
        avg_episode_length = sum(episode_lengths) / len(episode_lengths)
        avg_final_reward = sum(final_rewards) / len(final_rewards)
        
        return success_rate, avg_episode_length, avg_final_reward


def extract_environment_context_minimal(env_class: Optional[type] = None) -> dict:
    """Return lightweight context cues for reward generation without instantiating envs."""

    if env_class is None:
        return {
            "environment_class": "DroneEnv",
            "state": "state vector contains position (x,y,z), velocity (x,y,z), orientation quaternion, angular velocity (x,y,z)",
            "available_attributes": [
                "position", "velocity", "orientation", "angular_velocity",
                "accumulated_rotation", "flip_command", "target_position",
                "collision_vector", "flip_progress"
            ],
            "notes": "Use provided tensors; no rotation matrix attribute exists."
        }

    name = env_class.__name__

    if name == "FlipEnv":
        return {
            "environment_class": "FlipEnv",
            "state": "state observation is 60-dim (relative_pos, velocity, angular_velocity, orientation, target_pos, quat_diff, future_quat_diff)",
            "available_attributes": [
                "position", "velocity", "orientation", "angular_velocity",
                "is_collision", "collision_dis", "collision_vector", "target", "progress_buf", "command_quat",
                "command_quat_diff", "relative_pos", "hover_before_steps",
                "flip_steps", "hover_after_steps", "target_angle_total"
            ],
            "notes": (
                "Use 'target' (not 'target_position') to access the target position tensor. "
                "Use 'command_quat_diff' (4D tensor) to track rotation error. "
                "Use 'progress_buf' to track current step. "
                "Use 'relative_pos' for position error. "
                "Flip progress can be computed as: (progress_buf - hover_before_steps) / flip_steps if in flip phase, clamped to [0, 1]. "
                "Use 'collision_dis' (tensor of shape [num_envs]) for collision distance and 'collision_vector' (tensor of shape [num_envs, 3]) for collision direction. "
                "Use 'is_collision' (boolean tensor) to check if collision occurred."
            )
        }

    return {
        "environment_class": name,
        "notes": "Use tensors exposed on the environment (position, velocity, orientation, angular_velocity, collision_vector)."
    }


def extract_environment_context(env: Any) -> dict:
    """Extract relevant context information from environment."""
    context = {
        "environment_class": env.__class__.__name__,
        "num_agents": getattr(env, 'num_agent', 1),
        "observation_space": str(env.observation_space) if hasattr(env, 'observation_space') else None,
        "action_space": str(env.action_space) if hasattr(env, 'action_space') else None,
        "max_episode_steps": getattr(env, 'max_episode_steps', 256),
        "device": str(env.device) if hasattr(env, 'device') else "cuda",
        "sensors": [],
    }
    
    # Extract sensor information
    if hasattr(env, 'sensor_kwargs'):
        for sensor_config in env.sensor_kwargs:
            context["sensors"].append({
                "type": sensor_config.get("sensor_type"),
                "uuid": sensor_config.get("uuid"),
                "resolution": sensor_config.get("resolution")
            })
    
    return context

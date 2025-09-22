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


def extract_environment_context_minimal() -> dict:
    """Extract environment context without creating environment instances."""
    return {
        "environment_class": "NavigationEnv",
        "num_agents": 4,
        "observation_space": "MultiDict with 'state', 'depth', 'target'",
        "action_space": "Box(4,) for bodyrate control",
        "max_episode_steps": 256,
        "device": "cuda",
        "sensors": [
            {
                "type": "DEPTH",
                "uuid": "depth", 
                "resolution": [64, 64]
            }
        ],
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
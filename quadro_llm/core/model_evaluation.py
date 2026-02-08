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

"""
Training Utilities for VisFly-Eureka Integration

This module provides utilities for training VisFly environments with injected reward functions,
supporting both BPTT (differentiable simulation) and standard RL algorithms.
"""

import torch
import logging
from typing import Any, Optional, Union
from dataclasses import dataclass
import time


@dataclass
class TrainingResult:
    """Results from training a reward function."""
    success_rate: float
    episode_length: float
    training_time: float
    final_reward: float
    convergence_step: int
    reward_code: str = ""
    identifier: str = ""
    
    def score(self) -> float:
        """
        Calculate a composite score for ranking reward functions.
        
        Higher scores are better. This combines success rate and efficiency.
        """
        # Primary focus on success rate, secondary on episode efficiency
        efficiency_bonus = max(0, (256 - self.episode_length) / 256) * 0.3
        return self.success_rate * 0.7 + efficiency_bonus


def train_with_generated_reward(
    env: Any,
    reward_code: str,
    algorithm: str = "bptt",
    steps: int = 10000,
    device: str = "cuda"
) -> Optional[Any]:
    """
    Train a VisFly environment using a generated reward function.
    
    This function uses VisFly's native training systems (BPTT or PPO) to train
    policies with the injected reward function.
    
    Args:
        env: VisFly environment instance with injected reward
        reward_code: Reward function code (for logging/reference)
        algorithm: Training algorithm ("bptt" or "ppo")
        steps: Number of training steps
        device: Device for training
        
    Returns:
        Trained model or None if training failed
    """
    logger = logging.getLogger(__name__)
    
    try:
        if algorithm.lower() == "bptt":
            return _train_with_bptt(env, steps, device)
        elif algorithm.lower() == "ppo":
            return _train_with_ppo(env, steps, device)
        else:
            logger.error(f"Unknown algorithm: {algorithm}")
            return None
            
    except Exception as e:
        logger.error(f"Training failed with {algorithm}: {e}")
        return None


def _train_with_bptt(env: Any, steps: int, device: str) -> Optional[Any]:
    """Train using VisFly's BPTT algorithm."""
    logger = logging.getLogger(__name__)
    
    try:
        # Import VisFly BPTT
        from VisFly.utils.algorithms.BPTT import BPTT
        
        # Configure BPTT model
        model = BPTT(
            env=env,
            policy="MultiInputPolicy",
            device=device,
            learning_rate=3e-4,
            verbose=0  # Reduce output during training
        )
        
        logger.info(f"Starting BPTT training for {steps} steps")
        
        # Train the model
        model.learn(total_timesteps=steps)
        
        logger.info("BPTT training completed successfully")
        return model
        
    except ImportError:
        logger.error("VisFly BPTT algorithm not available")
        return None
    except Exception as e:
        logger.error(f"BPTT training failed: {e}")
        return None


def _train_with_ppo(env: Any, steps: int, device: str) -> Optional[Any]:
    """Train using Stable Baselines3 PPO."""
    logger = logging.getLogger(__name__)
    
    try:
        # Try to import VisFly's PPO first, fallback to stable-baselines3
        try:
            from VisFly.utils.algorithms.PPO import PPO
            ppo_class = PPO
            logger.debug("Using VisFly PPO")
        except ImportError:
            from stable_baselines3 import PPO
            ppo_class = PPO
            logger.debug("Using Stable Baselines3 PPO")
        
        # Configure PPO model
        model = ppo_class(
            "MultiInputPolicy",
            env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            verbose=0,
            device=device
        )
        
        logger.info(f"Starting PPO training for {steps} steps")
        
        # Train the model
        model.learn(total_timesteps=steps)
        
        logger.info("PPO training completed successfully")
        return model
        
    except ImportError as e:
        logger.error(f"PPO not available: {e}")
        return None
    except Exception as e:
        logger.error(f"PPO training failed: {e}")
        return None


def evaluate_model_performance(
    model: Any,
    env: Any,
    num_episodes: int = 10,
    max_episode_steps: int = 256
) -> TrainingResult:
    """
    Evaluate a trained model's performance.
    
    Args:
        model: Trained model
        env: Environment for evaluation
        num_episodes: Number of evaluation episodes
        max_episode_steps: Maximum steps per episode
        
    Returns:
        TrainingResult with performance metrics
    """
    logger = logging.getLogger(__name__)
    
    successes = 0
    episode_lengths = []
    final_rewards = []
    start_time = time.time()
    
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        step_count = 0
        episode_reward = 0
        
        while not done and step_count < max_episode_steps:
            # Get action from model
            if hasattr(model, 'predict'):
                # Stable Baselines3 interface
                action, _ = model.predict(obs, deterministic=True)
            elif hasattr(model, 'get_action'):
                # VisFly BPTT interface
                action = model.get_action(obs)
            else:
                # Fallback - random action
                action = env.action_space.sample()
            
            obs, reward, done, info = env.step(action)
            
            # Handle tensor rewards
            if isinstance(reward, torch.Tensor):
                episode_reward += reward.mean().item()
            else:
                episode_reward += reward
                
            step_count += 1
        
        # Check success condition
        success = False
        if hasattr(env, 'get_success'):
            try:
                success_tensor = env.get_success()
                success = success_tensor.any() if isinstance(success_tensor, torch.Tensor) else bool(success_tensor)
            except:
                success = episode_reward > 0  # Fallback success criteria
        else:
            success = episode_reward > 0
            
        if success:
            successes += 1
            
        episode_lengths.append(step_count)
        final_rewards.append(episode_reward)
    
    # Calculate metrics
    success_rate = successes / num_episodes
    avg_episode_length = sum(episode_lengths) / len(episode_lengths)
    avg_final_reward = sum(final_rewards) / len(final_rewards)
    evaluation_time = time.time() - start_time
    
    result = TrainingResult(
        success_rate=success_rate,
        episode_length=avg_episode_length,
        training_time=evaluation_time,
        final_reward=avg_final_reward,
        convergence_step=0  # Not applicable for evaluation
    )
    
    logger.info(f"Evaluation complete: {success_rate:.3f} success rate, {avg_episode_length:.1f} avg steps")
    return result


def create_training_config(algorithm: str, env_info: dict) -> dict:
    """
    Create training configuration based on environment and algorithm.
    
    Args:
        algorithm: Training algorithm name
        env_info: Environment information dictionary
        
    Returns:
        Training configuration dictionary
    """
    base_config = {
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "device": env_info.get("device", "cuda"),
        "verbose": 0
    }
    
    if algorithm.lower() == "bptt":
        base_config.update({
            "policy": "MultiInputPolicy",
            # BPTT-specific parameters can be added here
        })
    elif algorithm.lower() == "ppo":
        base_config.update({
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.01,
        })
    
    return base_config


def safe_model_training(env: Any, reward_code: str, algorithm: str, steps: int) -> Optional[Any]:
    """
    Safely train a model with comprehensive error handling.
    
    This function wraps the training process with error handling to ensure
    that training failures don't crash the optimization process.
    
    Args:
        env: VisFly environment with injected reward
        reward_code: Reward function code
        algorithm: Training algorithm
        steps: Training steps
        
    Returns:
        Trained model or None if training failed
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Validate environment state
        env.reset()
        initial_reward = env.get_reward()
        
        if not isinstance(initial_reward, torch.Tensor):
            logger.error("Environment reward function not working correctly")
            return None
        
        # Train with timeout protection
        start_time = time.time()
        timeout = 3600  # 1 hour timeout
        
        model = train_with_generated_reward(env, reward_code, algorithm, steps)
        
        training_time = time.time() - start_time
        if training_time > timeout:
            logger.warning(f"Training took {training_time:.1f}s, may have timed out")
        
        return model
        
    except Exception as e:
        logger.error(f"Safe training failed: {e}")
        return None


class TrainingMonitor:
    """
    Monitor training progress and performance.
    
    This class provides utilities to track training metrics and detect
    convergence or failure conditions.
    """
    
    def __init__(self):
        self.training_history = []
        self.logger = logging.getLogger(__name__)
    
    def record_training(self, result: TrainingResult):
        """Record a training result."""
        self.training_history.append(result)
        self.logger.info(f"Recorded training result: score={result.score():.3f}")
    
    def get_best_result(self) -> Optional[TrainingResult]:
        """Get the best training result so far."""
        if not self.training_history:
            return None
        return max(self.training_history, key=lambda x: x.score())
    
    def get_average_performance(self) -> float:
        """Get average performance across all training runs."""
        if not self.training_history:
            return 0.0
        return sum(r.score() for r in self.training_history) / len(self.training_history)
    
    def detect_improvement_trend(self, window_size: int = 5) -> bool:
        """
        Detect if recent results show improvement.
        
        Args:
            window_size: Number of recent results to consider
            
        Returns:
            True if showing improvement trend
        """
        if len(self.training_history) < window_size:
            return True  # Not enough data to determine trend
            
        recent_results = self.training_history[-window_size:]
        scores = [r.score() for r in recent_results]
        
        # Simple trend detection: is average of second half > first half?
        mid = len(scores) // 2
        first_half_avg = sum(scores[:mid]) / len(scores[:mid]) if mid > 0 else 0
        second_half_avg = sum(scores[mid:]) / len(scores[mid:])
        
        return second_half_avg > first_half_avg
    
    def clear_history(self):
        """Clear training history."""
        self.training_history.clear()
        self.logger.info("Training history cleared")
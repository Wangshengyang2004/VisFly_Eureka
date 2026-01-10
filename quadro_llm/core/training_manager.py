"""
Training manager for VisFly environments.
"""

import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import torch
import numpy as np


class TrainingManager:
    """Manages training of VisFly environments with injected rewards."""
    
    def __init__(self, algorithm: str = "bptt", device: str = "cuda"):
        """
        Initialize training manager.
        
        Args:
            algorithm: Training algorithm ("bptt" or "ppo")
            device: Device for training
        """
        self.algorithm = algorithm
        self.device = device if torch.cuda.is_available() else "cpu"
        self.logger = logging.getLogger(__name__)
        
    def train_with_reward(
        self,
        env: Any,
        training_steps: int = 10000,
        learning_rate: float = 1e-3,
        horizon: int = 96,
        log_interval: int = 100,
        save_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Train environment with current reward function.
        
        Args:
            env: Environment with injected reward function
            training_steps: Number of training steps
            learning_rate: Learning rate for optimization
            horizon: BPTT horizon length
            log_interval: Logging interval
            save_path: Path to save trained model
            
        Returns:
            Training results dictionary
        """
        start_time = time.time()
        
        if self.algorithm == "bptt":
            results = self._train_bptt(
                env, training_steps, learning_rate, horizon, log_interval, save_path
            )
        elif self.algorithm == "ppo":
            results = self._train_ppo(
                env, training_steps, learning_rate, log_interval, save_path
            )
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
        
        results['training_time'] = time.time() - start_time
        return results
    
    def _train_bptt(
        self,
        env: Any,
        training_steps: int,
        learning_rate: float,
        horizon: int,
        log_interval: int,
        save_path: Optional[Path]
    ) -> Dict[str, Any]:
        """
        Train using Back-Propagation Through Time.
        
        Args:
            env: Environment with differentiable dynamics
            training_steps: Number of training steps
            learning_rate: Learning rate
            horizon: BPTT horizon
            log_interval: Logging interval
            save_path: Model save path
            
        Returns:
            Training results
        """
        try:
            from VisFly.utils.algorithms.BPTT import BPTT
            from VisFly.utils.policies import extractors
            
            # Create BPTT model
            model = BPTT(
                policy="MultiInputPolicy",
                env=env,
                learning_rate=learning_rate,
                horizon=horizon,
                device=self.device,
                verbose=1,
                tensorboard_log=str(save_path) if save_path else None,
                policy_kwargs={
                    "features_extractor_class": extractors.StateExtractor,
                    "features_extractor_kwargs": {
                        "net_arch": {
                            "state": {"layer": [128, 64], "bn": False, "ln": False},
                        },
                        "activation_fn": torch.nn.ReLU,
                    },
                    "net_arch": {"pi": [64, 64], "vf": [64, 64]},
                    "activation_fn": torch.nn.ReLU,
                }
            )
            
            # Training loop with metrics collection
            episode_rewards = []
            episode_lengths = []
            success_rates = []
            
            for step in range(0, training_steps, horizon):
                # Reset environment
                obs = env.reset()
                episode_reward = 0
                episode_length = 0
                
                # Rollout horizon
                for h in range(horizon):
                    action = model.predict(obs, deterministic=False)[0]
                    obs, reward, done, info = env.step(action)
                    
                    episode_reward += reward.mean().item()
                    episode_length += 1
                    
                    if done.any():
                        break
                
                # Compute gradients and update
                loss = model.train_step()
                
                # Record metrics
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                
                # Check success
                if hasattr(env, 'get_success'):
                    success = env.get_success().float().mean().item()
                    success_rates.append(success)
                
                # Log progress
                if step % log_interval == 0:
                    avg_reward = np.mean(episode_rewards[-10:]) if episode_rewards else 0
                    avg_success = np.mean(success_rates[-10:]) if success_rates else 0
                    self.logger.info(
                        f"Step {step}/{training_steps} - "
                        f"Reward: {avg_reward:.3f}, Success: {avg_success:.3f}"
                    )
            
            # Save model if requested
            if save_path:
                model.save(str(save_path / "model"))
            
            return {
                'episode_rewards': episode_rewards,
                'episode_lengths': episode_lengths,
                'success_rates': success_rates,
                'final_success_rate': np.mean(success_rates[-20:]) if success_rates else 0,
                'convergence_step': self._find_convergence(success_rates),
            }
            
        except Exception as e:
            self.logger.error(f"BPTT training error: {e}")
            return {
                'error': str(e),
                'episode_rewards': [],
                'success_rates': [],
                'final_success_rate': 0,
            }
    
    def _train_ppo(
        self,
        env: Any,
        training_steps: int,
        learning_rate: float,
        log_interval: int,
        save_path: Optional[Path]
    ) -> Dict[str, Any]:
        """
        Train using Proximal Policy Optimization.
        
        Args:
            env: Gymnasium-compatible environment
            training_steps: Total timesteps
            learning_rate: Learning rate
            log_interval: Logging interval
            save_path: Model save path
            
        Returns:
            Training results
        """
        try:
            from stable_baselines3 import PPO
            from stable_baselines3.common.callbacks import EvalCallback
            
            # Create PPO model
            model = PPO(
                "MultiInputPolicy",
                env,
                learning_rate=learning_rate,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                verbose=1,
                tensorboard_log=str(save_path) if save_path else None,
                device=self.device,
            )
            
            # Setup evaluation callback
            eval_callback = EvalCallback(
                env,
                best_model_save_path=str(save_path) if save_path else None,
                log_path=str(save_path) if save_path else None,
                eval_freq=1000,
                deterministic=True,
                render=False,
            )
            
            # Train model
            model.learn(
                total_timesteps=training_steps,
                callback=eval_callback,
                log_interval=log_interval,
            )
            
            # Evaluate final performance
            episode_rewards = []
            success_rates = []
            
            for _ in range(20):  # Evaluation episodes
                obs = env.reset()
                done = False
                episode_reward = 0
                
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, done, info = env.step(action)
                    episode_reward += reward
                
                episode_rewards.append(episode_reward)
                
                if hasattr(env, 'get_success'):
                    success = env.get_success().float().mean().item()
                    success_rates.append(success)
            
            return {
                'episode_rewards': episode_rewards,
                'success_rates': success_rates,
                'final_success_rate': np.mean(success_rates) if success_rates else 0,
                'convergence_step': None,  # PPO doesn't track this easily
            }
            
        except Exception as e:
            self.logger.error(f"PPO training error: {e}")
            return {
                'error': str(e),
                'episode_rewards': [],
                'success_rates': [],
                'final_success_rate': 0,
            }
    
    def _find_convergence(
        self,
        success_rates: list,
        window: int = 20,
        threshold: float = 0.9
    ) -> Optional[int]:
        """
        Find convergence step in success rates.
        
        Args:
            success_rates: List of success rates
            window: Window size for averaging
            threshold: Success threshold for convergence
            
        Returns:
            Step number where convergence occurred or None
        """
        if len(success_rates) < window:
            return None
        
        for i in range(window, len(success_rates)):
            window_avg = np.mean(success_rates[i-window:i])
            if window_avg >= threshold:
                return i
        
        return None
"""
VisFly-Eureka Native Integration - Main Controller

This module provides the core EurekaVisFly class that orchestrates LLM-powered
reward function optimization directly with VisFly environments.
"""

import torch
import logging
from typing import Type, Dict, Any, List, Optional
from dataclasses import dataclass
import time

from .llm_engine import LLMEngine  
from .reward_injection import inject_generated_reward
from .training_utils import train_with_generated_reward, TrainingResult


@dataclass
class OptimizationConfig:
    """Configuration for reward optimization process"""
    iterations: int = 5
    samples: int = 16
    training_steps: int = 10000
    algorithm: str = "bptt"  # "bptt" or "ppo"
    evaluation_episodes: int = 10
    success_threshold: float = 0.8
    timeout_per_iteration: int = 1800  # 30 minutes per iteration


class EurekaVisFly:
    """
    Main controller for LLM-powered reward optimization in VisFly environments.
    
    This class orchestrates the complete optimization pipeline:
    1. Generate reward functions using LLM
    2. Inject rewards directly into VisFly environments  
    3. Train policies using native VisFly algorithms
    4. Evaluate and rank reward functions
    5. Iteratively improve based on performance feedback
    """
    
    def __init__(
        self,
        env_class: Type,
        task_description: str,
        llm_config: Dict[str, Any],
        env_kwargs: Optional[Dict[str, Any]] = None,
        optimization_config: Optional[OptimizationConfig] = None,
        device: str = "cuda"
    ):
        """
        Initialize Eureka-VisFly controller.
        
        Args:
            env_class: VisFly environment class (e.g., NavigationEnv)
            task_description: Natural language description of the task
            llm_config: Configuration for LLM (model, api_key, etc.)
            env_kwargs: Arguments for environment initialization
            optimization_config: Configuration for optimization process
            device: Device for training ("cuda" or "cpu")
        """
        self.env_class = env_class
        self.task_description = task_description
        self.device = device
        self.env_kwargs = env_kwargs or {}
        self.config = optimization_config or OptimizationConfig()
        
        # Initialize LLM engine
        self.llm = LLMEngine(**llm_config)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Track optimization history
        self.optimization_history: List[List[TrainingResult]] = []
        self.best_reward_functions: List[str] = []
        
    def create_environment(self, requires_grad: bool = False) -> Any:
        """Create a VisFly environment instance with specified configuration."""
        env_kwargs = self.env_kwargs.copy()
        env_kwargs.update({
            "device": self.device,
            "requires_grad": requires_grad
        })
        return self.env_class(**env_kwargs)
        
    def optimize_rewards(
        self, 
        iterations: Optional[int] = None, 
        samples: Optional[int] = None
    ) -> List[TrainingResult]:
        """
        Run the complete reward optimization pipeline.
        
        Args:
            iterations: Number of optimization iterations (overrides config)
            samples: Number of reward function samples per iteration (overrides config)
            
        Returns:
            List of TrainingResult objects sorted by performance (best first)
        """
        iterations = iterations or self.config.iterations
        samples = samples or self.config.samples
        
        self.logger.info(f"Starting reward optimization: {iterations} iterations, {samples} samples each")
        self.logger.info(f"Task: {self.task_description}")
        
        all_results = []
        
        for iteration in range(iterations):
            self.logger.info(f"=== Iteration {iteration + 1}/{iterations} ===")
            
            try:
                # Generate reward function candidates
                reward_functions = self.generate_reward_candidates(samples, iteration)
                
                if not reward_functions:
                    self.logger.warning(f"No valid reward functions generated in iteration {iteration + 1}")
                    continue
                    
                # Evaluate each reward function
                iteration_results = []
                for idx, reward_code in enumerate(reward_functions):
                    self.logger.info(f"Evaluating reward function {idx + 1}/{len(reward_functions)}")
                    
                    try:
                        result = self.evaluate_reward_function(reward_code, f"iter{iteration}_sample{idx}")
                        if result:
                            iteration_results.append(result)
                            all_results.append(result)
                    except Exception as e:
                        self.logger.error(f"Error evaluating reward function {idx + 1}: {e}")
                        continue
                
                # Store iteration results
                self.optimization_history.append(iteration_results)
                
                # Update best reward functions
                if iteration_results:
                    best_in_iteration = max(iteration_results, key=lambda x: x.score())
                    self.best_reward_functions.append(best_in_iteration.reward_code)
                    self.logger.info(f"Best in iteration {iteration + 1}: {best_in_iteration.score():.3f}")
                
            except Exception as e:
                self.logger.error(f"Error in iteration {iteration + 1}: {e}")
                continue
        
        # Sort all results by performance
        all_results.sort(key=lambda x: x.score(), reverse=True)
        
        if all_results:
            self.logger.info(f"Optimization complete. Best reward score: {all_results[0].score():.3f}")
        else:
            self.logger.warning("No successful reward functions found during optimization")
            
        return all_results
    
    def generate_reward_candidates(self, samples: int, iteration: int, feedback: Optional[str] = None) -> List[str]:
        """Generate reward function candidates using LLM."""
        # Build context-aware prompt without creating environment (avoids blocking)
        context_info = self.extract_environment_context_minimal()
        if feedback is None:
            feedback = self.get_iteration_feedback(iteration)
        
        # Generate reward functions
        reward_functions = self.llm.generate_reward_functions(
            task_description=self.task_description,
            context_info=context_info,
            feedback=feedback,
            samples=samples,
            env_class=self.env_class
        )
        
        self.logger.info(f"Generated {len(reward_functions)} reward function candidates")
        return reward_functions
        
    def extract_environment_context_minimal(self) -> Dict[str, Any]:
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
    
    def extract_environment_context(self, env) -> Dict[str, Any]:
        """Extract relevant context information from environment."""
        context = {
            "environment_class": env.__class__.__name__,
            "num_agents": getattr(env, 'num_agent', 1),
            "observation_space": str(env.observation_space) if hasattr(env, 'observation_space') else None,
            "action_space": str(env.action_space) if hasattr(env, 'action_space') else None,
            "max_episode_steps": getattr(env, 'max_episode_steps', 256),
            "device": str(env.device) if hasattr(env, 'device') else self.device,
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
    
    def get_iteration_feedback(self, iteration: int) -> str:
        """Generate feedback string based on previous iteration results."""
        if iteration == 0 or not self.optimization_history:
            return "This is the first iteration. Focus on basic task completion."
            
        prev_results = self.optimization_history[-1]
        if not prev_results:
            return "Previous iteration had no successful results. Try simpler reward designs."
            
        best_prev = max(prev_results, key=lambda x: x.score())
        
        feedback_parts = [
            f"Previous best score: {best_prev.score():.3f}",
            f"Success rate: {best_prev.success_rate:.3f}",
            f"Average episode length: {best_prev.episode_length:.1f}",
        ]
        
        if best_prev.success_rate < 0.3:
            feedback_parts.append("Focus on improving task completion rate.")
        elif best_prev.episode_length > 200:
            feedback_parts.append("Try to reduce episode length while maintaining success.")
        else:
            feedback_parts.append("Good performance. Try to optimize further.")
            
        return " ".join(feedback_parts)
    
    def evaluate_reward_function(self, reward_code: str, identifier: str) -> Optional[TrainingResult]:
        """
        Evaluate a single reward function by training and testing.
        
        Args:
            reward_code: Generated reward function code
            identifier: Unique identifier for this evaluation
            
        Returns:
            TrainingResult object or None if evaluation failed
        """
        try:
            start_time = time.time()
            
            # Create environment for training
            requires_grad = (self.config.algorithm == "bptt")
            env = self.create_environment(requires_grad=requires_grad)
            
            # Inject reward function
            inject_generated_reward(env, reward_code)
            
            # Train model
            model = train_with_generated_reward(
                env=env,
                reward_code=reward_code,
                algorithm=self.config.algorithm,
                steps=self.config.training_steps
            )
            
            if not model:
                self.logger.warning(f"Training failed for {identifier}")
                return None
                
            # Evaluate trained model
            evaluation_env = self.create_environment(requires_grad=False)
            inject_generated_reward(evaluation_env, reward_code)
            
            success_rate, avg_episode_length, final_reward = self.evaluate_trained_model(
                model, evaluation_env, self.config.evaluation_episodes
            )
            
            training_time = time.time() - start_time
            
            result = TrainingResult(
                success_rate=success_rate,
                episode_length=avg_episode_length,
                training_time=training_time,
                final_reward=final_reward,
                convergence_step=self.config.training_steps,
                reward_code=reward_code,
                identifier=identifier
            )
            
            self.logger.info(f"{identifier}: score={result.score():.3f}, success={success_rate:.3f}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error evaluating reward function {identifier}: {e}")
            return None
    
    def evaluate_trained_model(self, model, env, num_episodes: int):
        """Evaluate a trained model's performance."""
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
    
    def get_best_reward_function(self) -> Optional[str]:
        """Get the best reward function found during optimization."""
        if not self.best_reward_functions:
            return None
        
        # Return the most recent best reward function
        return self.best_reward_functions[-1]
    
    def save_optimization_results(self, filepath: str):
        """Save optimization results to file."""
        results_data = {
            "task_description": self.task_description,
            "config": self.config.__dict__,
            "optimization_history": [
                [result.__dict__ for result in iteration_results] 
                for iteration_results in self.optimization_history
            ],
            "best_reward_functions": self.best_reward_functions
        }
        
        import json
        with open(filepath, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        self.logger.info(f"Optimization results saved to {filepath}")
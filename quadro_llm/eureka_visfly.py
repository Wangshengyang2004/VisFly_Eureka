"""
VisFly-Eureka Native Integration - Main Controller

This module provides the core EurekaVisFly class that orchestrates LLM-powered
reward function optimization directly with VisFly environments.
"""

import torch
import logging
from typing import Type, Dict, Any, List, Optional
import time

from .llm.llm_engine import LLMEngine
from .utils.training_utils import TrainingResult
from .core.models import OptimizationConfig
from .core.model_evaluation import extract_environment_context_minimal
from .core.subprocess_evaluator import SubprocessRewardEvaluator
from .utils.tensorboard_utils import (
    load_tensorboard_logs,
    generate_eureka_style_feedback,
    extract_success_metric,
)


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
        device: str = "cuda",
        max_workers: int = 4,
        eval_env_config: Optional[Dict[str, Any]] = None,
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
        self.max_workers = max_workers
        self.eval_env_config = eval_env_config

        # Initialize LLM engine
        self.llm = LLMEngine(**llm_config)

        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # Initialize subprocess evaluator
        self.evaluator = SubprocessRewardEvaluator(self.logger)

        # Track optimization history
        self.optimization_history: List[List[TrainingResult]] = []
        self.best_reward_functions: List[str] = []

    def create_environment(self, requires_grad: bool = False) -> Any:
        """Create a VisFly environment instance with specified configuration."""
        env_kwargs = self.env_kwargs.copy()
        env_kwargs.update({"device": self.device, "requires_grad": requires_grad})
        return self.env_class(**env_kwargs)

    def optimize_rewards(
        self, iterations: Optional[int] = None, samples: Optional[int] = None
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

        # Starting optimization - details already logged by pipeline

        all_results = []

        for iteration in range(iterations):
            self.logger.info(f"Iteration {iteration + 1}")

            try:
                # Generate reward function candidates
                self.logger.info(f"Generating {samples} reward functions...")
                # Use enhanced feedback with tensorboard data if available
                feedback = (
                    self._generate_feedback_with_tensorboard(iteration)
                    if iteration > 0
                    else None
                )
                reward_functions = self.generate_reward_candidates(
                    samples, iteration, feedback
                )

                if not reward_functions:
                    self.logger.warning(
                        f"No valid reward functions generated in iteration {iteration + 1}"
                    )
                    continue

                # Save generated reward functions first
                self._save_reward_functions(reward_functions, iteration)

                # Use parallel evaluation with GPU task distribution
                identifiers = [
                    f"iter{iteration}_sample{idx}"
                    for idx in range(len(reward_functions))
                ]

                # Use max_workers from initialization
                max_concurrent = self.max_workers
                self.logger.info(
                    f"Using {max_concurrent} parallel workers for evaluation"
                )

                # Get the hydra output directory to create proper structure
                from hydra.core.hydra_config import HydraConfig

                if HydraConfig.initialized():
                    hydra_cfg = HydraConfig.get()
                    base_output_dir = hydra_cfg.runtime.output_dir
                else:
                    base_output_dir = None

                self.logger.info(f"Evaluating {len(reward_functions)} functions...")
                iteration_results = self.evaluator.evaluate_multiple_parallel(
                    reward_functions=reward_functions,
                    identifiers=identifiers,
                    env_config=self.env_kwargs,
                    optimization_config={
                        "algorithm": self.config.algorithm,
                        "evaluation_episodes": self.config.evaluation_episodes,
                    },
                    env_class_path=f"{self.env_class.__module__}.{self.env_class.__name__}",
                    max_concurrent=max_concurrent,
                    timeout=self.config.timeout_per_iteration,
                    base_output_dir=base_output_dir,
                    eval_env_config=self.eval_env_config,
                )

                # Convert to TrainingResult objects for compatibility
                training_results = []
                for result in iteration_results:
                    if result.training_successful:
                        training_result = TrainingResult(
                            success_rate=result.success_rate,
                            episode_length=result.episode_length,
                            training_time=result.training_time,
                            final_reward=result.final_reward,
                            convergence_step=result.convergence_step,
                            reward_code=result.reward_code,
                            identifier=result.identifier,
                        )
                        # Store log directory for tensorboard access
                        if hasattr(result, "log_dir"):
                            training_result.log_dir = result.log_dir
                        training_results.append(training_result)
                        all_results.append(training_result)

                # Store iteration results
                self.optimization_history.append(training_results)

                # Update best reward functions
                if training_results:
                    best_in_iteration = max(training_results, key=lambda x: x.score())
                    self.best_reward_functions.append(best_in_iteration.reward_code)
                    self.logger.info(f"Best score: {best_in_iteration.score():.3f}")

            except Exception as e:
                self.logger.error(f"Error in iteration {iteration + 1}: {e}")
                continue

        # Sort all results by performance
        all_results.sort(key=lambda x: x.score(), reverse=True)

        if all_results:
            self.logger.info(
                f"Optimization complete. Best reward score: {all_results[0].score():.3f}"
            )
        else:
            self.logger.warning(
                "No successful reward functions found during optimization"
            )

        return all_results

    def generate_reward_candidates(
        self, samples: int, iteration: int, feedback: Optional[str] = None
    ) -> List[str]:
        """Generate reward function candidates using LLM."""
        # Build context-aware prompt without creating environment (avoids blocking)
        context_info = extract_environment_context_minimal()
        if feedback is None:
            feedback = self.get_iteration_feedback(iteration)

        # Generate reward functions
        reward_functions = self.llm.generate_reward_functions(
            task_description=self.task_description,
            context_info=context_info,
            feedback=feedback,
            samples=samples,
            env_class=self.env_class,
        )

        return reward_functions

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
            feedback_parts.append(
                "Try to reduce episode length while maintaining success."
            )
        else:
            feedback_parts.append("Good performance. Try to optimize further.")

        return " ".join(feedback_parts)

    def _generate_feedback_with_tensorboard(self, iteration: int) -> str:
        """
        Generate enhanced feedback including tensorboard training curves.
        Like real Eureka, includes detailed training metrics.

        Args:
            iteration: Current iteration number

        Returns:
            Feedback string with tensorboard data for the LLM
        """
        if iteration == 0 or not self.optimization_history:
            return "This is the first iteration. Focus on basic task completion."

        # Get results from last iteration
        last_results = self.optimization_history[-1]

        if not last_results:
            return "Previous iteration had no successful results. Try simpler reward designs."

        # Sort by score and get best result
        best_result = max(last_results, key=lambda x: x.score())

        feedback_parts = []

        # Try to load tensorboard logs for best result
        if hasattr(best_result, "log_dir") and best_result.log_dir:
            try:
                tensorboard_logs = load_tensorboard_logs(best_result.log_dir)
                if tensorboard_logs:
                    # Generate Eureka-style feedback with training curves
                    tensorboard_feedback = generate_eureka_style_feedback(
                        tensorboard_logs
                    )
                    feedback_parts.append(tensorboard_feedback)
                    feedback_parts.append("")  # Add spacing
            except Exception as e:
                self.logger.warning(f"Could not load tensorboard logs: {e}")

        # If no tensorboard data, fall back to basic metrics
        if not feedback_parts:
            feedback_parts.append(f"Previous best score: {best_result.score():.3f}")
            feedback_parts.append(f"Success rate: {best_result.success_rate:.3f}")
            feedback_parts.append(
                f"Average episode length: {best_result.episode_length:.1f}"
            )

        # Add interpretation and suggestions based on metrics
        if best_result.success_rate < 0.3:
            feedback_parts.append(
                "\nThe reward function needs major improvements. Consider:"
            )
            feedback_parts.append("- Stronger distance-based rewards for navigation")
            feedback_parts.append("- Better balanced collision penalties")
            feedback_parts.append("- More effective reward shaping for exploration")
        elif best_result.success_rate < 0.7:
            feedback_parts.append(
                "\nThe reward function shows promise. To improve further:"
            )
            feedback_parts.append(
                "- Fine-tune the coefficients based on the training curves above"
            )
            feedback_parts.append("- If episode_length is high, add efficiency bonuses")
            feedback_parts.append(
                "- If convergence is slow, increase reward magnitudes"
            )
        else:
            feedback_parts.append("\nExcellent performance! For further optimization:")
            feedback_parts.append(
                "- Focus on reducing episode_length while maintaining success"
            )
            feedback_parts.append(
                "- Add smoothness penalties if trajectories are erratic"
            )
            feedback_parts.append("- Consider robustness across different scenarios")

        return "\n".join(feedback_parts)

    def _save_reward_functions(self, reward_functions: List[str], iteration: int):
        """Save generated reward functions to disk for debugging and analysis"""
        import os
        from pathlib import Path

        # Use Hydra output directory if available
        base_dir = Path("generated_rewards")
        try:
            from hydra.core.hydra_config import HydraConfig

            if HydraConfig.initialized():
                hydra_cfg = HydraConfig.get()
                base_dir = Path(hydra_cfg.runtime.output_dir) / "generated_rewards"
        except:
            pass  # Fall back to current directory

        # Create iteration directory
        iter_dir = base_dir / f"iteration_{iteration}"
        iter_dir.mkdir(parents=True, exist_ok=True)

        for idx, reward_code in enumerate(reward_functions):
            func_file = iter_dir / f"reward_function_{idx:02d}.py"
            with open(func_file, "w") as f:
                f.write(f"# Iteration {iteration}, Function {idx}\n")
                f.write(f"# Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write(reward_code)

        self.logger.info(
            f"Saved {len(reward_functions)} reward functions to {iter_dir}"
        )

    def evaluate_reward_function(
        self, reward_code: str, identifier: str
    ) -> Optional[TrainingResult]:
        """
        Evaluate a single reward function using subprocess isolation.

        Args:
            reward_code: Generated reward function code
            identifier: Unique identifier for this evaluation

        Returns:
            TrainingResult object or None if evaluation failed
        """
        try:
            # Use subprocess evaluator for isolation
            result = self.evaluator.evaluate_reward_function(
                reward_code=reward_code,
                identifier=identifier,
                env_config=self.env_kwargs,
                optimization_config={
                    "algorithm": self.config.algorithm,
                    "training_steps": self.config.training_steps,
                    "evaluation_episodes": self.config.evaluation_episodes,
                },
                env_class_path=f"{self.env_class.__module__}.{self.env_class.__name__}",
                timeout=self.config.timeout_per_iteration,
            )

            if result.training_successful:
                training_result = TrainingResult(
                    success_rate=result.success_rate,
                    episode_length=result.episode_length,
                    training_time=result.training_time,
                    final_reward=result.final_reward,
                    convergence_step=result.convergence_step,
                    reward_code=result.reward_code,
                    identifier=result.identifier,
                )

                return training_result
            else:
                return None

        except Exception as e:
            self.logger.error(f"Error in subprocess evaluation for {identifier}: {e}")
            return None

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
            "best_reward_functions": self.best_reward_functions,
        }

        import json

        with open(filepath, "w") as f:
            json.dump(results_data, f, indent=2)

        self.logger.info(f"Optimization results saved to {filepath}")

    def cleanup(self):
        """Cleanup resources"""
        if hasattr(self, "evaluator"):
            self.evaluator.cleanup()

    def __del__(self):
        """Cleanup on deletion"""
        self.cleanup()

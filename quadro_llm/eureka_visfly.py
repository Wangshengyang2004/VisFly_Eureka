"""
VisFly-Eureka Native Integration - Main Controller

This module provides the core EurekaVisFly class that orchestrates LLM-powered
reward function optimization directly with VisFly environments.
"""

import logging
from typing import Type, Dict, Any, List, Optional
import time

try:
    from hydra.core.hydra_config import HydraConfig
except ImportError:
    HydraConfig = None

from .llm.llm_engine import LLMEngine
from .core.models import TrainingResult
from .core.models import OptimizationConfig
from .core.model_evaluation import extract_environment_context_minimal
from .core.subprocess_evaluator import SubprocessRewardEvaluator
from .utils.tensorboard_utils import (
    load_tensorboard_logs,
    generate_eureka_style_feedback,
    extract_success_metric,
    append_dataframe_to_feedback,
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
        env_kwargs: Dict[str, Any],
        optimization_config: OptimizationConfig,
        device: str,
        max_workers: int,
        eval_env_config: Dict[str, Any],
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
        self.env_kwargs = env_kwargs
        self.config = optimization_config
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
        
        # Track complete iteration results (successful + failed) for detailed feedback
        self.complete_iteration_results: List[List] = []
        self.best_reward_functions: List[str] = []

    def create_environment(self, requires_grad: bool = False) -> Any:
        """Create a VisFly environment instance with specified configuration."""
        env_kwargs = self.env_kwargs.copy()
        # Only update requires_grad - preserve environment's original device config
        env_kwargs.update({"requires_grad": requires_grad})
        # Don't override device - let environment use its configured device
        return self.env_class(**env_kwargs)

    def optimize_rewards(self) -> List[TrainingResult]:
        """
        Run the complete reward optimization pipeline using configured parameters.

        Returns:
            List of TrainingResult objects sorted by performance (best first)
        """
        iterations = self.config.iterations
        samples = self.config.samples

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
                
                # Save LLM conversations for this iteration
                if HydraConfig and HydraConfig.initialized():
                    hydra_cfg = HydraConfig.get()
                    self.llm.save_conversations(hydra_cfg.runtime.output_dir, iteration)

                # Use parallel evaluation with GPU task distribution
                # Use simpler per-sample identifiers (sample{idx}); iteration handled separately in evaluator path logic
                identifiers = [f"sample{idx}" for idx in range(len(reward_functions))]

                # Use max_workers from initialization
                max_concurrent = self.max_workers
                self.logger.debug(
                    f"Using {max_concurrent} parallel workers for evaluation"
                )

                # Get the hydra output directory to create proper structure
                if HydraConfig and HydraConfig.initialized():
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
                        "iteration": iteration,
                        "record_video": self.config.record_video,
                    },
                    env_class_path=f"{self.env_class.__module__}.{self.env_class.__name__}",
                    max_concurrent=max_concurrent,
                    timeout=self.config.timeout_per_iteration,
                    base_output_dir=base_output_dir,
                    eval_env_config=self.eval_env_config,
                )

                # Store complete results (successful + failed) for detailed feedback
                self.complete_iteration_results.append(iteration_results)

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
                        if getattr(result, "log_dir", None):
                            # TrainingResult.log_dir accepts Optional[str]
                            training_result.log_dir = result.log_dir  # type: ignore[attr-defined]
                        training_results.append(training_result)
                        all_results.append(training_result)

                # Store iteration results
                self.optimization_history.append(training_results)

                # Update best reward functions
                if training_results:
                    best_in_iteration = max(training_results, key=lambda x: x.success_rate)
                    self.best_reward_functions.append(best_in_iteration.reward_code)
                    self.logger.info(f"Best success rate: {best_in_iteration.success_rate:.3f}")
                else:
                    self.logger.warning(f"All samples failed in iteration {iteration + 1}. Regenerating...")
                    # Regenerate with simpler feedback when all fail
                    simple_feedback = "All previous attempts failed due to coding errors. Generate simpler, more robust reward functions with basic components only. Use simple torch operations and avoid complex logic."
                    retry_functions = self.generate_reward_candidates(
                        samples=samples, iteration=iteration, feedback=simple_feedback
                    )
                    self.logger.info(f"Regenerated {len(retry_functions)} simpler reward functions")
                    
                    # Save regenerated functions with retry suffix
                    self._save_reward_functions(retry_functions, iteration, suffix="_retry")
                    
                    # Re-evaluate the regenerated functions (simplified retry logic)
                    self.logger.info("Evaluating regenerated functions...")
                    # Skip retry evaluation for now to avoid infinite loops
                    # Could be implemented with reduced complexity requirements

            except Exception as e:
                self.logger.error(f"Error in iteration {iteration + 1}: {e}")
                continue

        # Sort all results by performance
        all_results.sort(key=lambda x: x.success_rate, reverse=True)

        if all_results:
            self.logger.info(
                f"Optimization complete. Best success rate: {all_results[0].success_rate:.3f}"
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
        # Use enhanced tensorboard feedback method with sample success/failure info
        return self._generate_feedback_with_tensorboard(iteration)

    def _generate_feedback_with_tensorboard(self, iteration: int, all_iteration_results=None) -> str:
        """
        Generate enhanced feedback including tensorboard training curves.
        Like real Eureka, includes detailed training metrics and sample success/failure info.

        Args:
            iteration: Current iteration number
            all_iteration_results: All results from the iteration (successful + failed)

        Returns:
            Feedback string with tensorboard data for the LLM
        """
        if iteration == 0 or not self.optimization_history:
            return "This is the first iteration. Focus on basic task completion."

        # Get results from last iteration
        last_results = self.optimization_history[-1]
        
        # Get complete results (successful + failed) from last iteration
        complete_last_results = []
        if self.complete_iteration_results and len(self.complete_iteration_results) >= iteration:
            complete_last_results = self.complete_iteration_results[iteration - 1]

        if not last_results:
            return "Previous iteration had no successful results. Try simpler reward designs."

        # Sort by score and get best result with sophisticated ranking
        def rank_result(result):
            """
            Rank training results with multi-criteria selection:
            1. Primary: Success rate (higher is better)
            2. Secondary: Episode length (longer is better when success rates are equal)
            """
            return (result.success_rate, result.episode_length)
        
        best_result = max(last_results, key=rank_result)
        
        # Log the selection reasoning and find the best result's index
        all_success_rates = [r.success_rate for r in last_results]
        best_result_index = next(
            i for i, r in enumerate(last_results) 
            if r.identifier == best_result.identifier
        )
        
        if len(set(all_success_rates)) == 1:  # All same success rate
            self.logger.info(
                f"All samples have same success rate ({all_success_rates[0]:.3f}). "
                f"Selected sample {best_result_index} ('{best_result.identifier}') with longest episode length: {best_result.episode_length:.1f}"
            )
        else:
            self.logger.info(
                f"Selected sample {best_result_index} ('{best_result.identifier}') with best success rate: {best_result.success_rate:.3f}"
            )

        # Classify samples into successful vs failed
        successful_samples = []
        failed_samples = []
        
        if complete_last_results:
            for i, result in enumerate(complete_last_results):
                if result.training_successful:
                    successful_samples.append(i)
                else:
                    failed_samples.append(i)
        
        feedback_parts = []
        
        # Add sample success/failure overview
        if complete_last_results:
            feedback_parts.append(f"ITERATION RESULTS: {len(complete_last_results)} samples evaluated")
            if successful_samples:
                feedback_parts.append(f"Successful samples: {successful_samples}")
            if failed_samples:
                feedback_parts.append(f"Failed samples: {failed_samples}")
            feedback_parts.append("")  # Add spacing

        # Try to load tensorboard logs for best result
        if hasattr(best_result, "log_dir") and best_result.log_dir:
            try:
                tensorboard_logs = load_tensorboard_logs(best_result.log_dir)
                if tensorboard_logs:
                    # Generate Eureka-style feedback with training curves
                    tensorboard_feedback = generate_eureka_style_feedback(
                        tensorboard_logs
                    )
                    # Append DataFrame summary for next iteration agent
                    enhanced_feedback = append_dataframe_to_feedback(
                        tensorboard_feedback, tensorboard_logs,
                        selected_index=best_result_index,
                        total_candidates=len(last_results)
                    )
                    feedback_parts.append(enhanced_feedback)
                    feedback_parts.append("")  # Add spacing
            except Exception as e:
                self.logger.warning(f"Could not load tensorboard logs: {e}")

        # If no tensorboard data, fall back to basic metrics
        if len(feedback_parts) <= 3:  # Only sample overview added, no TensorBoard data
            if not complete_last_results:
                # Add sample info if not already added
                feedback_parts.append(f"ITERATION RESULTS: {len(last_results)} samples evaluated")
                feedback_parts.append(f"Successful samples: {list(range(len(last_results)))}")
                feedback_parts.append(f"Failed samples: []")
                feedback_parts.append("")
            
            feedback_parts.append(f"SELECTED REWARD FUNCTION: #{best_result_index} (from {len(last_results)} successful candidates)")
            feedback_parts.append(f"Previous best success rate: {best_result.success_rate:.3f}")
            feedback_parts.append(f"Success rate: {best_result.success_rate:.3f}")
            feedback_parts.append(
                f"Average episode length: {best_result.episode_length:.1f}"
            )
            
            # Add selection reasoning when using episode length as tiebreaker
            if len(set(all_success_rates)) == 1 and len(last_results) > 1:
                feedback_parts.append(
                    f"\nSelection reason: All {len(last_results)} samples had equal success rate ({all_success_rates[0]:.3f}). "
                    f"Reward function #{best_result_index} was selected for achieving the longest episode length "
                    f"({best_result.episode_length:.1f}), indicating better survival/learning."
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

    def _save_reward_functions(self, reward_functions: List[str], iteration: int, suffix: str = ""):
        """Save generated reward functions to disk for debugging and analysis"""
        import os
        from pathlib import Path

        # Use Hydra output directory if available
        base_dir = Path("generated_rewards")
        try:
            if HydraConfig and HydraConfig.initialized():
                hydra_cfg = HydraConfig.get()
                base_dir = Path(hydra_cfg.runtime.output_dir) / "generated_rewards"
        except Exception as e:
            self.logger.debug(f"Hydra not available or not initialized: {e}")  # Fall back to current directory

        # Create iteration directory
        iter_dir_name = f"iter{iteration}{suffix}"
        iter_dir = base_dir / iter_dir_name
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
                    # choose a conservative per-sample training steps; worker also supports defaults
                    "training_steps": 10000,
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

"""
VisFly-Eureka Native Integration - Main Controller

This module provides the core EurekaVisFly class that orchestrates LLM-powered
reward function optimization directly with VisFly environments.
"""

import logging
import numpy as np
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
from .core.elite_voter import EliteVoter
from .utils.tensorboard_utils import (
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
        env_kwargs: Dict[str, Any],
        optimization_config: OptimizationConfig,
        device: str,
        max_workers: int,
        eval_env_config: Dict[str, Any],
        use_coefficient_tuning: bool = False,
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
        
        # Initialize elite voter for LLM-based selection
        self.elite_voter = EliteVoter(self.llm)

        # Track optimization history
        self.optimization_history: List[List[TrainingResult]] = []
        
        # Track complete iteration results (successful + failed) for detailed feedback
        self.complete_iteration_results: List[List] = []
        self.best_reward_functions: List[str] = []
        
        # Track elite voter results for feedback generation
        self.elite_vote_results: List = []  # Store EliteVoterResult for each iteration
        
        # Coefficient tuning mode flag
        self.use_coefficient_tuning = use_coefficient_tuning

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
                # Generate feedback from previous iteration results
                # This feedback will be used for generating candidates AND for updating history later
                feedback = (
                    self._generate_feedback(iteration)
                    if iteration > 0
                    else None
                )
                reward_functions = self.generate_reward_candidates(
                    samples, iteration, feedback, use_coefficient_tuning=self.use_coefficient_tuning
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
                        "gpu_memory_requirement_mb": self.config.gpu_memory_requirement_mb,
                    },
                    env_class_path=f"{self.env_class.__module__}.{self.env_class.__name__}",
                    max_concurrent=max_concurrent,
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

                # Select best reward function using elite voter
                if iteration_results:
                    vote_result = self.elite_voter.vote(iteration_results)
                    best_result = iteration_results[vote_result.selected_index]
                    self.best_reward_functions.append(best_result.reward_code)
                    self.elite_vote_results.append(vote_result)
                    self.logger.info(f"Elite voter selected: {best_result.identifier} (success_rate={best_result.success_rate:.3f}, confidence={vote_result.confidence:.2f})")
                    self.logger.debug(f"Voter reasoning: {vote_result.reasoning}")
                    
                    # Validate vote_result consistency
                    if vote_result.selected_index >= len(iteration_results):
                        self.logger.error(
                            f"Iteration {iteration + 1}: vote_result.selected_index ({vote_result.selected_index}) "
                            f"out of range for iteration_results (len={len(iteration_results)})"
                        )
                    elif iteration_results[vote_result.selected_index].identifier != best_result.identifier:
                        self.logger.warning(
                            f"Iteration {iteration + 1}: vote_result.selected_index points to different identifier. "
                            f"Expected: {best_result.identifier}, "
                            f"Got: {iteration_results[vote_result.selected_index].identifier}"
                        )
                    
                    # Update conversation history with elite reward function only
                    # Reconstruct the user prompt that was used for this iteration
                    # Note: For history, we include static info only if it's the first iteration
                    # (subsequent iterations already have it in history)
                    # IMPORTANT: Use the feedback generated at the START of this iteration,
                    # not regenerated here, because self.optimization_history already contains
                    # current iteration results which would cause wrong data to be retrieved
                    context_info = extract_environment_context_minimal(self.env_class)
                    prev_feedback = feedback if iteration > 0 else ""
                    prev_elite_code = self.best_reward_functions[-2] if len(self.best_reward_functions) >= 2 else None
                    
                    from .llm.prompts import create_user_prompt, extract_env_code_without_reward
                    env_code = extract_env_code_without_reward(self.env_class)
                    # Only include static info for first iteration (iteration 0)
                    include_static_info = iteration == 0
                    user_prompt = create_user_prompt(
                        task_description=self.task_description,
                        context_info=context_info,
                        feedback=prev_feedback,
                        env_code=env_code,
                        api_doc=self.llm.api_doc_content if self.llm.api_doc_content else None,
                        human_reward_code=None,  # Not needed for history update
                        elite_reward_code=prev_elite_code,
                        include_static_info=include_static_info,
                    )
                    
                    # Update history with elite reward function (only elite, not first candidate)
                    self.llm._update_conversation_history(best_result.reward_code, user_prompt)
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
        self, samples: int, iteration: int, feedback: Optional[str] = None, use_coefficient_tuning: bool = False
    ) -> List[str]:
        """Generate reward function candidates using LLM."""
        if use_coefficient_tuning:
            # Coefficient tuning mode: LLM only outputs coefficients
            if feedback is None:
                feedback = self.get_iteration_feedback(iteration)
            
            # Get current best coefficients for reference
            current_coefficients = None
            if iteration > 0 and self.optimization_history:
                # Extract coefficients from best result if available
                # For now, we'll just pass None and let LLM generate fresh
                pass
            
            reward_functions = self.llm.generate_coefficients(
                task_description=self.task_description,
                feedback=feedback,
                samples=samples,
                current_coefficients=current_coefficients,
            )
        else:
            # Standard Eureka mode: LLM generates full reward function
            # Build context-aware prompt without creating environment (avoids blocking)
            context_info = extract_environment_context_minimal(self.env_class)
            if feedback is None:
                feedback = self.get_iteration_feedback(iteration)

            # Get elite reward code from previous iteration if available
            elite_reward_code = None
            if iteration > 0 and self.best_reward_functions:
                elite_reward_code = self.best_reward_functions[-1]
            
            # Generate reward functions
            # Note: We don't update conversation history here - it will be updated
            # after elite voter selection with only the elite reward function
            reward_functions = self.llm.generate_reward_functions(
                task_description=self.task_description,
                context_info=context_info,
                feedback=feedback,
                samples=samples,
                env_class=self.env_class,
                previous_elite_reward=elite_reward_code,
            )

        return reward_functions

    def get_iteration_feedback(self, iteration: int) -> str:
        """Generate feedback string based on previous iteration results."""
        return self._generate_feedback(iteration)

    def _generate_feedback(self, iteration: int, all_iteration_results=None) -> str:
        """
        Generate concise feedback focused on evaluation results (LaRes-style).
        TensorBoard data is handled by elite voter, not included in feedback.

        Args:
            iteration: Current iteration number
            all_iteration_results: All results from the iteration (successful + failed)

        Returns:
            Concise feedback string with evaluation metrics for the LLM
        """
        # Check if we have previous iteration results
        if iteration == 0:
            return "This is the first iteration. Focus on basic task completion."

        # optimization_history stores successful TrainingResults; complete_iteration_results has everything
        if not self.optimization_history and not self.complete_iteration_results:
            self.logger.warning(
                f"Iteration {iteration}: No optimization history or complete results available. "
                "Returning first-iteration feedback."
            )
            return "This is the first iteration. Focus on basic task completion."

        # Get results from last iteration (prefer optimization_history, fallback to complete_iteration_results)
        # For iteration N, we need results from iteration N-1
        # optimization_history[iteration-1] = results from iteration iteration-1
        if self.optimization_history and len(self.optimization_history) >= iteration:
            last_results = self.optimization_history[iteration - 1]
            self.logger.info(
                f"Using optimization_history[iteration-1] ({iteration-1}) with {len(last_results)} samples"
            )
        elif self.optimization_history:
            # Fallback: use last available results (should not happen in normal flow)
            self.logger.warning(
                f"Iteration {iteration}: optimization_history has {len(self.optimization_history)} entries, "
                f"expected at least {iteration}. Using last available."
            )
            last_results = self.optimization_history[-1]
        elif self.complete_iteration_results and len(self.complete_iteration_results) >= iteration:
            # Use complete results (includes both successful and failed)
            self.logger.info(
                f"Using complete_iteration_results as fallback, "
                f"iteration {iteration-1} has {len(self.complete_iteration_results[iteration-1])} samples"
            )
            # Extract TrainingResult from RewardFunctionResult
            complete_last_results = self.complete_iteration_results[iteration - 1]
            # Convert RewardFunctionResult to TrainingResult for compatibility
            last_results = [
                TrainingResult(
                    success_rate=r.success_rate,
                    episode_length=r.episode_length,
                    training_time=r.training_time,
                    final_reward=r.final_reward,
                    convergence_step=r.convergence_step,
                    reward_code=r.reward_code,
                    identifier=r.identifier,
                    log_dir=getattr(r, 'log_dir', None),  # Preserve log_dir for TensorBoard feedback
                )
                for r in complete_last_results
                if r.training_successful  # Only include training-completed samples
            ]
        else:
            self.logger.warning(
                f"Iteration {iteration}: No results available in optimization_history or complete_iteration_results"
            )
            return "This is the first iteration. Focus on basic task completion."

        # Get complete results (successful + failed) from last iteration for detailed feedback
        complete_last_results = []
        if self.complete_iteration_results and len(self.complete_iteration_results) >= iteration:
            complete_last_results = self.complete_iteration_results[iteration - 1]

        if not last_results:
            self.logger.warning(f"Iteration {iteration}: No training results available for feedback")
            return "Previous iteration had no successful results. Try simpler reward designs."

        # Get elite voter result if available
        # For iteration N, we need the vote_result from iteration N-1
        # elite_vote_results[0] = vote_result for iteration 0, etc.
        vote_result = None
        if iteration > 0 and self.elite_vote_results and len(self.elite_vote_results) >= iteration:
            vote_result = self.elite_vote_results[iteration - 1]
            self.logger.debug(
                f"Iteration {iteration}: Retrieved vote_result from iteration {iteration - 1}, "
                f"selected_index={vote_result.selected_index}, confidence={vote_result.confidence:.2f}"
            )
        
        # Use elite voter selection if available, otherwise fallback to heuristic
        best_result = None
        best_result_index = None
        best_result_raw = None
        
        if vote_result and complete_last_results:
            # Get the selected identifier from the vote result
            # We need to find which result was selected in the previous iteration
            # vote_result.selected_index is based on the iteration_results list from that iteration
            selected_identifier = None
            if vote_result.selected_index < len(complete_last_results):
                candidate_result = complete_last_results[vote_result.selected_index]
                selected_identifier = candidate_result.identifier
            else:
                # Index out of range - fallback to first successful
                self.logger.warning(
                    f"Iteration {iteration}: vote_result.selected_index ({vote_result.selected_index}) "
                    f"out of range for complete_last_results (len={len(complete_last_results)}). "
                    "Using first successful result."
                )
            
            # Find by identifier for reliability
            if selected_identifier:
                best_result_raw = next(
                    (r for r in complete_last_results if r.identifier == selected_identifier),
                    None
                )
                if best_result_raw is None:
                    self.logger.warning(
                        f"Iteration {iteration}: Could not find result with identifier '{selected_identifier}'. "
                        "Falling back to index-based lookup."
                    )
                    # Fallback to index
                    if vote_result.selected_index < len(complete_last_results):
                        best_result_raw = complete_last_results[vote_result.selected_index]
                    else:
                        best_result_raw = None
            else:
                # Fallback: use first successful result
                best_result_raw = next(
                    (r for r in complete_last_results if r.training_successful),
                    None
                )
            
            if best_result_raw:
                # Find corresponding TrainingResult
                best_result = next(
                    (r for r in last_results if r.identifier == best_result_raw.identifier),
                    None
                )
                if best_result:
                    best_result_index = next(
                        i for i, r in enumerate(last_results) 
                        if r.identifier == best_result.identifier
                    )
                    self.logger.info(
                        f"Using elite voter selection: {best_result.identifier} "
                        f"(success_rate={best_result.success_rate:.3f}, confidence={vote_result.confidence:.2f})"
                    )
                else:
                    # Log detailed information for debugging
                    available_identifiers = [r.identifier for r in last_results]
                    self.logger.warning(
                        f"Iteration {iteration}: Elite voter selected '{best_result_raw.identifier}' "
                        f"but not found in last_results. "
                        f"Available identifiers in last_results: {available_identifiers}. "
                        f"best_result_raw.training_successful={best_result_raw.training_successful}. "
                        "Using fallback."
                    )
                    best_result = None
            else:
                self.logger.warning(
                    f"Iteration {iteration}: Could not find elite voter selected result. Using fallback."
                )
                best_result = None
        
        # Fallback to heuristic if no vote result
        if best_result is None:
            def rank_result(result):
                return (result.success_rate, result.episode_length)
            best_result = max(last_results, key=rank_result)
            best_result_index = next(
                i for i, r in enumerate(last_results) 
                if r.identifier == best_result.identifier
            )
            self.logger.info(
                f"Fallback: Selected {best_result.identifier} with success_rate={best_result.success_rate:.3f}"
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
        
        # Get best result's evaluation summary (LaRes-style feedback)
        # TensorBoard data is handled by elite voter, not included in feedback
        # best_result is already selected by elite voter (or fallback heuristic)
        # best_result_raw is already set if elite voter was used, otherwise find it
        if best_result_raw is None and complete_last_results:
            # Fallback: find the raw result matching best_result
            for r in complete_last_results:
                if r.training_successful and r.identifier == best_result.identifier:
                    best_result_raw = r
                    break
        
        # LaRes-style concise feedback
        feedback_parts.append(
            f"Based on the above reward function, the current RL policy's win rate is {best_result.success_rate:.3f}."
        )
        
        # Add elite voter's analysis if available
        if vote_result and vote_result.reasoning:
            feedback_parts.append("")
            feedback_parts.append("Elite Voter Analysis (from previous iteration selection):")
            feedback_parts.append(vote_result.reasoning)
            feedback_parts.append("")
        
        # Add evaluation metrics from evaluation_summary if available
        if best_result_raw and best_result_raw.evaluation_summary:
            eval_sum = best_result_raw.evaluation_summary
            feedback_parts.append(
                f"Below are the scores of the current policy on different metrics across multiple rounds during the evaluation process:"
            )
            
            eval_metrics = []
            if 'success_count' in eval_sum:
                eval_metrics.append(f"Success: {eval_sum['success_count']}/{eval_sum.get('actual_evaluation_episodes', 'N/A')} episodes")
            if 'mean_episode_length' in eval_sum:
                eval_metrics.append(f"Episode Length: {eval_sum['mean_episode_length']:.1f}")
            if 'mean_final_distance' in eval_sum and not np.isnan(eval_sum.get('mean_final_distance', np.nan)):
                eval_metrics.append(f"Final Distance to Target: {eval_sum['mean_final_distance']:.3f}m")
            if 'collision_count' in eval_sum:
                eval_metrics.append(f"Collisions: {eval_sum['collision_count']}")
            if 'mean_episode_reward' in eval_sum:
                eval_metrics.append(f"Episode Reward: {eval_sum['mean_episode_reward']:.2f}")
            
            if eval_metrics:
                feedback_parts.append(" ".join(eval_metrics))
            
            # Add per-episode statistics if available (similar to LaRes's all_infos)
            if best_result_raw.episode_statistics:
                feedback_parts.append("\nPer-episode evaluation results:")
                for i, ep_stat in enumerate(best_result_raw.episode_statistics[:5]):  # Show first 5 episodes
                    ep_parts = [f"Episode {i+1}:"]
                    if ep_stat.get('success'):
                        ep_parts.append("Success")
                    else:
                        ep_parts.append("Failed")
                    if 'final_distance_to_target' in ep_stat:
                        ep_parts.append(f"Distance: {ep_stat['final_distance_to_target']:.3f}m")
                    if 'collision' in ep_stat:
                        ep_parts.append(f"Collision: {ep_stat['collision']}")
                    feedback_parts.append(" ".join(ep_parts))
        
        feedback_parts.append("")
        feedback_parts.append("Please carefully analyze the policy feedback. Some helpful tips for analyzing the policy feedback:")
        feedback_parts.append("    (1) If the success rates are always near zero, then you must rewrite the entire reward function")
        feedback_parts.append("    (2) If the current policy has already performed well on certain metrics, the focus should shift to the subsequent tasks.")
        feedback_parts.append("    (3) If the reward is excessively large, it may need to be appropriately scaled to avoid learning issues.")

        # Add interpretation and suggestions based on metrics
        if best_result.success_rate < 0.3:
            feedback_parts.append(
                "\nThe reward function needs major improvements. Consider:"
            )
            feedback_parts.append("- Stronger rewards for progress toward task objective")
            feedback_parts.append("- Better balanced penalties for constraint violations")
            feedback_parts.append("- More effective reward shaping to guide learning")
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

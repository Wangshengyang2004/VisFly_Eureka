"""
Main optimization pipeline for VisFly-Eureka.

This module contains the EurekaPipeline class that orchestrates
the complete reward optimization workflow for any environment.
"""

import time
import logging
import json
from pathlib import Path
from typing import List

from .eureka_visfly import EurekaVisFly
from .core.models import OptimizationReport
from .utils.gpu_monitor import GPUMonitor, DynamicGPUResourceManager


class EurekaPipeline:
    """
    General purpose pipeline for reward optimization using Eureka methodology.

    This pipeline implements the complete Eureka workflow:
    1. Iterative reward function generation with LLM
    2. Direct injection into VisFly environments
    3. Training with comprehensive logging
    4. Result analysis and ranking
    5. Best function selection
    """

    def __init__(
        self,
        eureka_controller: EurekaVisFly,
        output_dir: str = "./eureka_output",
    ):
        """
        Initialize the production pipeline.

        Args:
            eureka_controller: Pre-configured EurekaVisFly instance
            output_dir: Directory for saving results
        """
        self.eureka = eureka_controller
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Setup logging
        self.logger = logging.getLogger("EurekaPipeline")

        # Setup artifacts directory
        self.artifacts_dir = self.output_dir / "artifacts"
        self.artifacts_dir.mkdir(exist_ok=True, parents=True)

        # Initialize GPU monitoring and resource management
        self.gpu_monitor = GPUMonitor(update_interval=5.0)
        self.gpu_resource_manager = DynamicGPUResourceManager(self.gpu_monitor)



    def run_optimization(self) -> OptimizationReport:
        """
        Run the complete optimization pipeline using configured parameters.

        Returns:
            OptimizationReport with comprehensive results and analysis
        """
        start_time = time.time()

        try:
            # Run iterative optimization
            results = self.eureka.optimize_rewards()

            # Analyze results and create report
            final_report = self._create_optimization_report(
                results, time.time() - start_time
            )

            # Save outputs
            self._save_outputs(final_report)

            self.logger.info(
                f"Optimization completed in {final_report.execution_time:.1f}s"
            )

            return final_report

        finally:
            pass

    def _create_optimization_report(
        self, results: List, execution_time: float
    ) -> OptimizationReport:
        """Create optimization report from results"""

        # Calculate metrics
        successful_results = [r for r in results if r.success_rate >= 0]
        total_samples = len(results)
        successful_samples = len(successful_results)

        # Best performance
        best_performance = {}
        if successful_results:
            best = successful_results[0]  # Results are already sorted by score
            best_performance = {
                "success_rate": best.success_rate,
                "episode_length": best.episode_length,
                "training_time": best.training_time,
                "final_reward": best.final_reward,
                "score": best.success_rate,
            }

        # Create iteration history from eureka optimization history
        iteration_history = []
        for i, iter_results in enumerate(self.eureka.optimization_history):
            if iter_results:
                best_in_iter = max(iter_results, key=lambda x: x.success_rate)
                best_idx = iter_results.index(best_in_iter)
                
                # Convert TrainingResult to RewardFunctionResult for IterationSummary
                from .core.models import RewardFunctionResult, IterationSummary
                reward_results = []
                for result in iter_results:
                    reward_result = RewardFunctionResult(
                        reward_code=result.reward_code,
                        identifier=result.identifier,
                        training_successful=True,  # Only successful results are in the history
                        success_rate=result.success_rate,
                        episode_length=result.episode_length,
                        training_time=result.training_time,
                        final_reward=result.final_reward,
                        convergence_step=result.convergence_step,
                        error_message="",
                        log_dir=getattr(result, 'log_dir', None)
                    )
                    reward_results.append(reward_result)
                
                iteration_summary = IterationSummary(
                    iteration=i + 1,
                    samples=reward_results,
                    best_sample_idx=best_idx,
                    best_success_rate=best_in_iter.success_rate,
                    best_correlation=0.0,  # Not implemented
                    execution_rate=len(iter_results) / max(1, len(iter_results)),  # All successful
                    generation_time=0.0,  # Not tracked currently
                    total_training_time=sum(r.training_time for r in iter_results)
                )
                iteration_history.append(iteration_summary)

        return OptimizationReport(
            total_samples=total_samples,
            successful_samples=successful_samples,
            best_performance=best_performance,
            improvement_metrics={"baseline_comparison": "not_implemented"},
            execution_time=execution_time,
            output_directory=str(self.output_dir),
            iteration_history=iteration_history,
            best_reward_code=successful_results[0].reward_code
            if successful_results
            else None,
        )

    def _save_outputs(self, report: OptimizationReport):
        """Save optimization results"""

        # Save optimization report
        report_file = self.output_dir / "optimization_report.json"
        with open(report_file, "w") as f:
            json.dump(report.to_dict(), f, indent=2)

        # Save best reward function
        if report.best_reward_code:
            best_function_file = self.output_dir / "best_reward_function.py"
            with open(best_function_file, "w") as f:
                f.write(report.best_reward_code)

        self.logger.info(f"Results saved to {self.output_dir}")


def run_production_pipeline():
    """Run the complete production pipeline with default configuration

    Note: This function is for standalone testing only.
    Production usage should go through main.py with Hydra configuration.
    """
    raise NotImplementedError(
        "Standalone pipeline execution not supported. "
        "Please use main.py with Hydra configuration."
    )


if __name__ == "__main__":
    # Run the production pipeline
    results = run_production_pipeline()

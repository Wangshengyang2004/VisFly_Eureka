"""
Main optimization pipeline for VisFly-Eureka.

This module contains the EurekaPipeline class that orchestrates
the complete reward optimization workflow for any environment.
"""

import time
import logging
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional

from .eureka_visfly import EurekaVisFly
from .core.models import (
    IterationSummary,
    OptimizationReport,
    RewardFunctionResult,
)


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

        # GPU monitoring is handled by the SubprocessRewardEvaluator
        # No need for a separate GPU monitor in the pipeline



    def run_optimization(self) -> OptimizationReport:
        """
        Run the complete optimization pipeline using configured parameters.

        Returns:
            OptimizationReport with comprehensive results and analysis
        """
        start_time = time.time()

        # Run iterative optimization
        results = self.eureka.optimize_rewards()

        execution_time = time.time() - start_time

        # Gather detailed results for all successful samples
        successful_reward_results = self._gather_successful_reward_results()

        # Identify the best-performing reward across all successes
        best_reward_result = self._select_best_reward_result(
            successful_reward_results
        )

        # Analyze results and create report
        final_report = self._create_optimization_report(
            results, execution_time
        )

        # Create best_sample symlink pointing directly to train folder
        if best_reward_result:
            best_artifact_root = self._derive_artifact_root(best_reward_result)
            if best_artifact_root and best_artifact_root.exists():
                alias_path = self._ensure_best_alias(best_artifact_root)
                final_report.best_artifacts_dir = str(alias_path)
                self.logger.info(
                    "Best sample artifacts available at: %s",
                    alias_path,
                )

        # Save outputs
        self._save_outputs(final_report)

        self.logger.info(
            f"Optimization completed in {final_report.execution_time:.1f}s"
        )

        return final_report

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
        for iteration_index, iter_results in enumerate(
            self.eureka.optimization_history, start=1
        ):
            if not iter_results:
                continue

            reward_results = [
                RewardFunctionResult(
                    reward_code=result.reward_code,
                    identifier=result.identifier,
                    training_successful=True,
                    success_rate=result.success_rate,
                    episode_length=result.episode_length,
                    training_time=result.training_time,
                    final_reward=result.final_reward,
                    convergence_step=result.convergence_step,
                    error_message="",
                    log_dir=getattr(result, "log_dir", None),
                )
                for result in iter_results
            ]

            best_idx, best_result = max(
                enumerate(reward_results), key=lambda item: item[1].success_rate
            )

            iteration_history.append(
                IterationSummary(
                    iteration=iteration_index,
                    samples=reward_results,
                    best_sample_idx=best_idx,
                    best_success_rate=best_result.success_rate,
                    best_correlation=0.0,
                    execution_rate=1.0,
                    generation_time=0.0,
                    total_training_time=sum(r.training_time for r in iter_results),
                )
            )

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

    def _gather_successful_reward_results(self) -> List[RewardFunctionResult]:
        """Collect all successful reward function evaluations with artifacts."""

        successful_results: List[RewardFunctionResult] = []

        for iteration_results in getattr(self.eureka, "complete_iteration_results", []):
            for result in iteration_results:
                if getattr(result, "training_successful", False):
                    successful_results.append(result)

        return successful_results

    def _select_best_reward_result(
        self, results: List[RewardFunctionResult]
    ) -> Optional[RewardFunctionResult]:
        """Return the highest-scoring reward function result from the provided list."""

        best_result: Optional[RewardFunctionResult] = None

        for result in results:
            if best_result is None:
                best_result = result
                continue

            if result.success_rate > best_result.success_rate:
                best_result = result
                continue

            if (
                result.success_rate == best_result.success_rate
                and result.episode_length > best_result.episode_length
            ):
                best_result = result

        return best_result


    def _ensure_best_alias(self, target_path: Path) -> Path:
        """Expose a stable best_sample directory pointing to target_path."""

        alias_path = self.output_dir / "best_sample"

        if alias_path.exists() or alias_path.is_symlink():
            if alias_path.is_symlink() or alias_path.is_file():
                alias_path.unlink()
            else:
                shutil.rmtree(alias_path)

        try:
            alias_path.symlink_to(target_path, target_is_directory=True)
        except Exception:
            shutil.copytree(target_path, alias_path, dirs_exist_ok=True)

        return alias_path

    def _derive_artifact_root(
        self, result: RewardFunctionResult
    ) -> Optional[Path]:
        """Best-effort deduction of the evaluation artifact directory."""

        if result.video_paths:
            first_video = Path(result.video_paths[0])
            return first_video.parent.parent  # .../sampleX/videos -> sampleX

        if result.log_dir:
            log_path = Path(result.log_dir)
            return log_path.parent

        return None


def run_production_pipeline():
    """Run the complete production pipeline with default configuration

    Note: This function is for standalone testing only.
    Production usage should go through main.py with Hydra configuration.
    """
    raise NotImplementedError(
        "Standalone pipeline execution not supported. "
        "Please use main.py with Hydra configuration."
    )


# Backwards compatible alias for older tests/imports.
Pipeline = EurekaPipeline


if __name__ == "__main__":
    # Run the production pipeline
    results = run_production_pipeline()

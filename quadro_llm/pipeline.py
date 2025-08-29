"""
Main optimization pipeline for VisFly-Eureka.

This module contains the EurekaNavigationPipeline class that orchestrates
the complete reward optimization workflow.
"""

import os
import time
import logging
import json
from pathlib import Path
from typing import Any, List

from .eureka_visfly import EurekaVisFly
from .core.models import OptimizationConfig, OptimizationReport
from .training.parallel_training import ParallelTrainingManager
from .utils.gpu_monitor import GPUMonitor, DynamicGPUResourceManager


class EurekaNavigationPipeline:
    """
    Production pipeline for NavigationEnv reward optimization using Eureka methodology.

    This pipeline implements the complete Eureka workflow:
    1. Iterative reward function generation with LLM
    2. Direct injection into VisFly NavigationEnv
    3. Training with comprehensive logging
    4. Result analysis and ranking
    5. Best function selection
    """

    def __init__(
        self,
        task_description: str,
        config: Any,  # DictConfig from Hydra
        output_dir: str = "./eureka_output",
        env_class=None,
    ):
        """
        Initialize the production pipeline.

        Args:
            task_description: Natural language description of navigation task
            config: Hydra DictConfig containing all configuration
            output_dir: Directory for saving results
            env_class: Optional environment class override
        """
        self.task_description = task_description
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Setup logging
        self.setup_logging()

        # Store env class if provided
        self.env_class = env_class

        # Setup artifacts directory
        self.artifacts_dir = self.output_dir / "artifacts"
        self.artifacts_dir.mkdir(exist_ok=True, parents=True)

        # Initialize GPU monitoring and resource management
        self.gpu_monitor = GPUMonitor()
        self.gpu_resource_manager = DynamicGPUResourceManager(self.gpu_monitor)

        # Initialize parallel training manager
        self.training_manager = ParallelTrainingManager(
            results_dir=str(self.output_dir),
            gpu_resource_manager=self.gpu_resource_manager,
        )

        # Initialize components
        self._initialize_pipeline()

        # Pipeline initialized

    def setup_logging(self):
        """Setup logging (uses existing logging configuration)"""
        log_level = getattr(logging, self.config.logging.level.upper())

        # Create logger that inherits from root logger
        self.logger = logging.getLogger("EurekaNavigationPipeline")
        self.logger.setLevel(log_level)

        # Disable propagation to prevent duplicate console output
        self.logger.propagate = False

    def _initialize_pipeline(self):
        """Initialize pipeline components"""
        try:
            # Use provided environment class or default to NavigationEnv
            if self.env_class is None:
                from VisFly.envs.NavigationEnv import NavigationEnv

                self.env_class = NavigationEnv

            # Convert env config to dict for passing to environment
            from omegaconf import OmegaConf

            env_kwargs = OmegaConf.to_container(self.config.envs.env, resolve=True)
            env_kwargs["device"] = self.config.execution.device

            # Convert target to tensor if present
            if "target" in env_kwargs:
                import torch

                env_kwargs["target"] = torch.tensor([env_kwargs["target"]])

            # Load API keys for LLM
            api_keys_path = Path(__file__).parent.parent / "configs" / "api_keys.yaml"
            import yaml

            with open(api_keys_path, "r") as f:
                api_config = yaml.safe_load(f)
                vendor = self.config.llm.llm.vendor
                vendor_config = api_config["providers"][vendor]

            llm_config = {
                "model": self.config.llm.llm.model,
                "api_key": vendor_config["api_key"],
                "base_url": vendor_config.get("base_url"),
                "temperature": self.config.llm.llm.temperature,
                "max_tokens": self.config.llm.llm.max_tokens,
                "timeout": self.config.llm.llm.timeout,
                "max_retries": self.config.llm.llm.max_retries,
            }

            # Create optimization config
            opt_config = OptimizationConfig(
                iterations=self.config.optimization.iterations,
                samples=self.config.optimization.samples,
                algorithm=self.config.optimization.algorithm,
                evaluation_episodes=self.config.optimization.evaluation_episodes,
                timeout_per_iteration=self.config.execution.timeout_per_sample,
            )

            # Get eval env config if available
            eval_env_config = None
            if "eval_env" in self.config.envs:
                eval_env_config = OmegaConf.to_container(
                    self.config.envs.eval_env, resolve=True
                )
                eval_env_config["device"] = self.config.execution.device
                # Convert target to tensor if present
                if "target" in eval_env_config:
                    import torch

                    eval_env_config["target"] = torch.tensor(
                        [eval_env_config["target"]]
                    )

            # Create Eureka controller
            self.eureka = EurekaVisFly(
                env_class=self.env_class,
                task_description=self.task_description,
                llm_config=llm_config,
                env_kwargs=env_kwargs,
                optimization_config=opt_config,
                device=self.config.execution.device,
                max_workers=self.config.execution.max_workers,
                eval_env_config=eval_env_config,
            )

            # Initialization complete - log at debug level only

        except Exception as e:
            self.logger.error(f"Failed to initialize pipeline: {e}")
            raise

    def run_optimization(self) -> OptimizationReport:
        """
        Run the complete optimization pipeline.

        Returns:
            OptimizationReport with comprehensive results and analysis
        """
        start_time = time.time()

        # Start optimization

        try:
            # Start parallel training manager
            self.training_manager.start()

            # Run iterative optimization
            results = self.eureka.optimize_rewards(
                iterations=self.config.optimization.iterations,
                samples=self.config.optimization.samples,
            )

            # Analyze results and create report
            final_report = self._create_optimization_report(
                results, time.time() - start_time
            )

            # Save outputs
            self._save_outputs(final_report)

            self.logger.info(
                f"Optimization completed in {final_report.execution_time:.1f}s"
            )
            self.logger.info("=" * 60)

            return final_report

        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {e}")
            raise
        finally:
            # Stop parallel training manager
            self.training_manager.stop()

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
                "score": best.score(),
            }

        # Create iteration history from eureka optimization history
        iteration_history = []
        for i, iter_results in enumerate(self.eureka.optimization_history):
            if iter_results:
                best_in_iter = max(iter_results, key=lambda x: x.score())
                iteration_history.append(
                    {
                        "iteration": i + 1,
                        "best_success_rate": best_in_iter.success_rate,
                        "num_samples": len(iter_results),
                        "successful_samples": len(
                            [r for r in iter_results if r.success_rate >= 0]
                        ),
                    }
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

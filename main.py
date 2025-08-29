"""
VisFly-Eureka Main Entry Point

This is the main entry point for running VisFly-Eureka reward optimization.
Uses Hydra for configuration management, supporting multiple environments,
LLM providers, and optimization settings.

Usage:
    python main.py                                    # Use default config
    python main.py llm=gpt4o_openai                  # Use OpenAI GPT-4o
    python main.py env=racing_env task=racing_task    # Racing optimization
    python main.py optimization.iterations=10         # Override iterations
    python main.py system=high_performance            # High performance mode
"""

import os
import sys
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import torch

# Suppress verbose third-party logging
logging.getLogger("numexpr").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Add project paths
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "VisFly"))

from quadro_llm import EurekaNavigationPipeline, OptimizationReport


def setup_logging(cfg: DictConfig):
    """Setup logging configuration"""
    log_level = getattr(logging, cfg.logging.level.upper())

    # Only use console output - Hydra captures it to .hydra/main.log automatically
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )

    logger = logging.getLogger(__name__)
    logger.info("Logging setup complete")
    return logger


def print_configuration_summary(cfg: DictConfig, logger: logging.Logger):
    """Print a summary of the current configuration"""

    logger.info("=" * 60)
    logger.info("VISFLY-EUREKA OPTIMIZATION")
    logger.info("=" * 60)
    logger.info(f"Pipeline: {cfg.pipeline.name}")
    logger.info(f"Task: {cfg.task.task.name} ({cfg.task.task.category})")
    logger.info(f"Environment: {cfg.optimization.environment}")
    logger.info(f"LLM: {cfg.llm.llm.vendor}/{cfg.llm.llm.model}")
    logger.info(f"Algorithm: {cfg.optimization.algorithm.upper()}")
    logger.info(f"Device: {cfg.execution.device.upper()}")
    logger.info(
        f"Optimization: {cfg.optimization.iterations} iterations × {cfg.optimization.samples} samples"
    )
    logger.info(f"Output: {cfg.pipeline.output_dir}")
    logger.info("=" * 60)


def run_optimization(cfg: DictConfig, logger: logging.Logger) -> OptimizationReport:
    """Run the complete optimization pipeline"""

    # Use Hydra's working directory for outputs
    from hydra.core.hydra_config import HydraConfig

    if HydraConfig.initialized():
        hydra_cfg = HydraConfig.get()
        hydra_output_dir = hydra_cfg.runtime.output_dir
        logger.info(f"Using Hydra output directory: {hydra_output_dir}")
    else:
        hydra_output_dir = os.getcwd()
        logger.info(
            f"Hydra not initialized, using current directory: {hydra_output_dir}"
        )

    # Initialize pipeline with config directly
    pipeline = EurekaNavigationPipeline(
        task_description=cfg.task.task.description,
        config=cfg,
        output_dir=hydra_output_dir,
    )

    # Run optimization
    logger.info("Starting optimization pipeline...")
    results = pipeline.run_optimization()

    return results


def print_results_summary(results: OptimizationReport, logger: logging.Logger):
    """Print optimization results summary"""

    logger.info("=" * 60)
    logger.info("OPTIMIZATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Total Samples: {results.total_samples}")
    success_pct = (
        (results.successful_samples / results.total_samples * 100)
        if results.total_samples > 0
        else 0.0
    )
    logger.info(
        f"Successful Samples: {results.successful_samples} ({success_pct:.1f}%)"
    )

    if results.best_performance:
        logger.info(
            f"Best Success Rate: {results.best_performance.get('success_rate', 0):.3f}"
        )
        logger.info(
            f"Best Episode Length: {results.best_performance.get('episode_length', 0):.1f}"
        )
        logger.info(f"Best Score: {results.best_performance.get('score', 0):.4f}")

    if results.improvement_metrics:
        improvement = results.improvement_metrics.get("success_rate_improvement", 0)
        relative_improvement = (
            results.improvement_metrics.get("relative_improvement", 0) * 100
        )
        logger.info(
            f"Success Rate Improvement: {improvement:+.3f} ({relative_improvement:+.1f}%)"
        )

    logger.info(f"Execution Time: {results.execution_time:.1f}s")
    logger.info("=" * 60)

    # Iteration progression
    if results.iteration_history:
        logger.info("Iteration Progression:")
        for iter_data in results.iteration_history:
            logger.info(
                f"  Iter {iter_data['iteration']}: "
                f"Success={iter_data['best_success_rate']:.3f}, "
                f"Exec Rate={iter_data['execution_rate']:.1%}"
            )


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main entry point for VisFly-Eureka optimization.

    Args:
        cfg: Hydra configuration object
    """

    # Setup logging
    logger = setup_logging(cfg)

    try:
        # Print working directory only
        logger.info(f"Working directory: {os.getcwd()}")

        # Print summary
        print_configuration_summary(cfg, logger)

        # Run optimization
        results = run_optimization(cfg, logger)

        # Print results
        print_results_summary(results, logger)

        # Success message
        logger.info("🎉 VisFly-Eureka optimization completed successfully!")
        logger.info(f"📊 Results saved to: {os.getcwd()}")

    except Exception as e:
        logger.error(f"❌ Optimization failed: {e}")
        raise

    except KeyboardInterrupt:
        logger.info("⚠️  Optimization interrupted by user")
        raise


if __name__ == "__main__":
    main()

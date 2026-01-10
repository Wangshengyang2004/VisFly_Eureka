"""
CLI-facing logging helpers.

These functions keep `main.py` thin and focused on orchestration.
"""

from __future__ import annotations

import logging
from pathlib import Path

from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from .core.models import OptimizationReport

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def setup_logging(cfg: DictConfig) -> logging.Logger:
    """Configure process-wide logging for the CLI entrypoint."""
    log_level = getattr(logging, cfg.logging.level.upper())

    # Only configure if not already configured (prevents duplicate handlers)
    if not logging.getLogger().hasHandlers():
        # Only use console output - Hydra captures it to .hydra/main.log automatically.
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler()],
        )

    logger = logging.getLogger(__name__)
    logger.info("Logging setup complete")
    return logger


def print_configuration_summary(cfg: DictConfig, logger: logging.Logger) -> None:
    """Log a summary of the current configuration."""
    logger.info("=" * 60)
    logger.info("VISFLY-EUREKA OPTIMIZATION")
    logger.info("=" * 60)
    logger.info("Pipeline: %s", cfg.pipeline.name)
    if hasattr(cfg, "mode") and hasattr(cfg.mode, "name"):
        logger.info("Mode: %s", cfg.mode.name)

    # Get values from task config
    env_name = OmegaConf.select(cfg, "task.task.envs")
    task_name = OmegaConf.select(cfg, "task.task.name")
    task_category = OmegaConf.select(cfg, "task.task.category")

    logger.info("Task: %s (%s)", task_name, task_category)
    logger.info("Environment: %s", env_name)
    logger.info("LLM: %s/%s", cfg.llm.llm.vendor, cfg.llm.llm.model)
    logger.info("Algorithm: %s", str(cfg.optimization.algorithm).upper())
    logger.info("Device: %s", str(cfg.execution.device).upper())
    logger.info(
        "Optimization: %s iterations × %s samples",
        cfg.optimization.iterations,
        cfg.optimization.samples,
    )
    logger.info("Configured output root: %s", cfg.pipeline.output_dir)
    if HydraConfig.initialized():
        logger.info("Hydra run directory: %s", HydraConfig.get().runtime.output_dir)
    logger.info("=" * 60)


def print_results_summary(results: OptimizationReport, logger: logging.Logger) -> None:
    """Log optimization results summary."""
    logger.info("=" * 60)
    logger.info("OPTIMIZATION RESULTS")
    logger.info("=" * 60)
    logger.info("Total Samples: %s", results.total_samples)

    success_pct = (
        (results.successful_samples / results.total_samples * 100)
        if results.total_samples > 0
        else 0.0
    )
    logger.info(
        "Successful Samples: %s (%.1f%%)",
        results.successful_samples,
        success_pct,
    )

    if results.best_performance:
        logger.info(
            "Best Success Rate: %.3f",
            results.best_performance.get("success_rate", 0),
        )
        logger.info(
            "Best Episode Length: %.1f",
            results.best_performance.get("episode_length", 0),
        )
        logger.info("Best Score: %.4f", results.best_performance.get("score", 0))

    if results.improvement_metrics:
        improvement = results.improvement_metrics.get("success_rate_improvement", 0)
        relative_improvement = results.improvement_metrics.get("relative_improvement", 0) * 100
        logger.info(
            "Success Rate Improvement: %+0.3f (%+0.1f%%)",
            improvement,
            relative_improvement,
        )

    logger.info("Execution Time: %.1fs", results.execution_time)
    logger.info("=" * 60)

    if results.iteration_history:
        logger.info("Iteration Progression:")
        for iter_data in results.iteration_history:
            logger.info(
                "  Iter %s: Success=%.3f, Exec Rate=%.1f%%",
                getattr(iter_data, "iteration", None),
                getattr(iter_data, "best_success_rate", 0.0),
                getattr(iter_data, "execution_rate", 0.0) * 100.0,
            )


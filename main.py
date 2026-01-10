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

import logging
import os
import sys
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

# Suppress verbose third-party logging
logging.getLogger("numexpr").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Add project paths
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "VisFly"))

from quadro_llm import OptimizationReport
from quadro_llm.bootstrap import create_mode
from quadro_llm.cli_logging import (
    print_configuration_summary,
    print_results_summary,
    setup_logging,
)


def run_optimization(cfg: DictConfig, logger: logging.Logger) -> OptimizationReport:
    """Run the complete optimization pipeline"""

    # Use Hydra's working directory for outputs
    if HydraConfig.initialized():
        hydra_cfg = HydraConfig.get()
        hydra_output_dir = hydra_cfg.runtime.output_dir
        logger.info(f"Using Hydra output directory: {hydra_output_dir}")
    else:
        hydra_output_dir = os.getcwd()
        logger.info(
            f"Hydra not initialized, using current directory: {hydra_output_dir}"
        )

    mode = create_mode(cfg, logger, output_dir=hydra_output_dir)
    logger.info("Starting optimization (%s)...", getattr(mode, "name", "unknown"))
    return mode.run()


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

        # Success message (no emoji)
        logger.info("VisFly-Eureka optimization completed successfully.")
        logger.info("Results saved to: %s", results.output_directory)
        if results.best_artifacts_dir:
            logger.info(
                "Best sample artifacts (videos, plots, logs) stored at: %s",
                results.best_artifacts_dir,
            )

    except Exception as e:
        logger.error("Optimization failed: %s", e)
        raise

    except KeyboardInterrupt:
        logger.info("Optimization interrupted by user")
        raise


if __name__ == "__main__":
    main()

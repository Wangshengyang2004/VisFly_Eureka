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

from quadro_llm import EurekaPipeline, EurekaVisFly, OptimizationConfig, OptimizationReport


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


def load_environment_class(env_name: str):
    """Load environment class by name"""
    env_registry = {
        "navigation": ("VisFly.envs.NavigationEnv", "NavigationEnv"),
        "hover": ("VisFly.envs.HoverEnv", "HoverEnv"),
        "racing": ("VisFly.envs.RacingEnv", "RacingEnv"),
        "tracking": ("VisFly.envs.ObjectTrackingEnv", "ObjectTrackingEnv"),
    }
    
    if env_name not in env_registry:
        raise ValueError(f"Unknown environment: {env_name}")
    
    module_name, class_name = env_registry[env_name]
    import importlib
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def prepare_env_config(env_config: DictConfig) -> dict:
    """Convert environment config to dict and handle tensor conversions"""
    from omegaconf import OmegaConf
    import torch
    
    env_kwargs = OmegaConf.to_container(env_config, resolve=True)
    
    # Convert target to tensor if present
    if "target" in env_kwargs:
        env_kwargs["target"] = torch.tensor([env_kwargs["target"]])
    
    return env_kwargs


def load_llm_config(cfg: DictConfig) -> dict:
    """Load and prepare LLM configuration with API keys
    
    Note: All LLM parameters come from config files (configs/llm/*.yaml)
    No default values - config is always correct and complete.
    """
    import yaml
    
    api_keys_path = PROJECT_ROOT / "configs" / "api_keys.yaml"
    
    with open(api_keys_path, "r") as f:
        api_config = yaml.safe_load(f)
        vendor = cfg.llm.llm.vendor
        vendor_config = api_config["providers"][vendor]

    # All values come from config - no defaults needed
    llm_config = {
        "model": cfg.llm.llm.model,
        "api_key": vendor_config["api_key"], 
        "base_url": vendor_config["base_url"],
        "temperature": cfg.llm.llm.temperature,
        "max_tokens": cfg.llm.llm.max_tokens,
        "timeout": cfg.llm.llm.timeout,
        "max_retries": cfg.llm.llm.max_retries,
    }
    
    # Add thinking configuration if present in config
    if hasattr(cfg.llm.llm, 'thinking') and hasattr(cfg.llm.llm.thinking, 'enabled'):
        llm_config["thinking_enabled"] = cfg.llm.llm.thinking.enabled
    
    # Add batching configuration if present in config
    if hasattr(cfg.llm.llm, 'batching'):
        batching = cfg.llm.llm.batching
        if hasattr(batching, 'strategy'):
            llm_config["batching_strategy"] = batching.strategy
        if hasattr(batching, 'supports_n_parameter'):
            llm_config["supports_n_parameter"] = batching.supports_n_parameter
        if hasattr(batching, 'max_concurrent'):
            llm_config["max_concurrent"] = batching.max_concurrent
    
    return llm_config


def create_optimization_config(cfg: DictConfig) -> OptimizationConfig:
    """Create optimization configuration from config"""
    return OptimizationConfig(
        iterations=cfg.optimization.iterations,
        samples=cfg.optimization.samples,
        algorithm=cfg.optimization.algorithm,
        evaluation_episodes=cfg.optimization.evaluation_episodes,
        timeout_per_iteration=cfg.execution.timeout_per_sample,
    )


def prepare_eval_env_config(cfg: DictConfig, base_env_config: dict) -> dict:
    """Prepare evaluation environment configuration"""
    if "eval_env" in cfg.envs:
        eval_env_config = prepare_env_config(cfg.envs.eval_env)
    else:
        eval_env_config = base_env_config.copy()
    
    return eval_env_config


def create_eureka_controller(cfg: DictConfig, logger: logging.Logger) -> EurekaVisFly:
    """Create and configure EurekaVisFly controller from config"""
    
    # Load environment class
    env_class = load_environment_class(cfg.optimization.environment)
    
    # Prepare configurations
    env_kwargs = prepare_env_config(cfg.envs.env)
    llm_config = load_llm_config(cfg)
    opt_config = create_optimization_config(cfg)
    eval_env_config = prepare_eval_env_config(cfg, env_kwargs)
    
    # Create Eureka controller  
    env_device = env_kwargs["device"]
    return EurekaVisFly(
        env_class=env_class,
        task_description=cfg.task.task.description,
        llm_config=llm_config,
        env_kwargs=env_kwargs,
        optimization_config=opt_config,
        device=env_device,
        max_workers=cfg.execution.max_workers,
        eval_env_config=eval_env_config,
    )


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

    # Create Eureka controller with complex initialization logic
    eureka_controller = create_eureka_controller(cfg, logger)
    
    # Initialize simplified pipeline
    pipeline = EurekaPipeline(
        eureka_controller=eureka_controller,
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
            # iter_data is IterationSummary dataclass
            logger.info(
                "  Iter %s: Success=%.3f, Exec Rate=%.1f%%",
                getattr(iter_data, 'iteration', None),
                getattr(iter_data, 'best_success_rate', 0.0),
                getattr(iter_data, 'execution_rate', 0.0) * 100.0,
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

        # Success message (no emoji)
        logger.info("VisFly-Eureka optimization completed successfully.")
        logger.info("Results saved to: %s", os.getcwd())

    except Exception as e:
        logger.error("Optimization failed: %s", e)
        raise

    except KeyboardInterrupt:
        logger.info("Optimization interrupted by user")
        raise


if __name__ == "__main__":
    main()

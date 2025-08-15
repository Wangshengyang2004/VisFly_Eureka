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

# Add project paths
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "VisFly"))

from eureka_visfly import EurekaNavigationPipeline, OptimizationReport


def setup_logging(cfg: DictConfig):
    """Setup logging configuration"""
    log_level = getattr(logging, cfg.logging.level.upper())
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('pipeline.log'),  # Save to current Hydra directory
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Logging setup complete")
    return logger


def validate_config(cfg: DictConfig, logger: logging.Logger):
    """Validate configuration parameters"""
    
    # Validate LLM configuration
    if not cfg.llm.llm.api_key:
        raise ValueError("LLM API key is required but not provided")
    
    if cfg.llm.llm.model not in ["gpt-4o", "gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]:
        logger.warning(f"Unusual model specified: {cfg.llm.llm.model}")
    
    # Validate optimization parameters
    if cfg.optimization.iterations <= 0:
        raise ValueError("optimization.iterations must be positive")
    
    if cfg.optimization.samples <= 0:
        raise ValueError("optimization.samples must be positive")
    
    if cfg.optimization.algorithm not in ["bptt", "ppo"]:
        raise ValueError("optimization.algorithm must be 'bptt' or 'ppo'")
    
    # Validate device settings
    if cfg.execution.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available, falling back to CPU")
        cfg.execution.device = "cpu"
    
    # Validate environment
    if not cfg.env.env.name:
        raise ValueError("Environment name is required")
        
    logger.info("Configuration validation passed")


def create_environment_config(cfg: DictConfig) -> dict:
    """Create environment configuration from Hydra config"""
    
    env_config = {
        "num_agent_per_scene": cfg.env.env.num_agent_per_scene,
        "num_scene": cfg.env.env.num_scene,
        "visual": cfg.env.env.visual,
        "requires_grad": cfg.env.env.requires_grad,
        "max_episode_steps": cfg.env.env.max_episode_steps,
        "device": cfg.execution.device,
    }
    
    # Add sensor configuration
    if "sensors" in cfg.env.env:
        sensor_kwargs = []
        for sensor_cfg in cfg.env.env.sensors:
            sensor_kwargs.append({
                "sensor_type": sensor_cfg.sensor_type,
                "uuid": sensor_cfg.uuid,
                "resolution": list(sensor_cfg.resolution)
            })
        env_config["sensor_kwargs"] = sensor_kwargs
    
    # Add target configuration
    if "target" in cfg.env.env:
        env_config["target"] = torch.tensor([cfg.env.env.target])
    
    # Add dynamics configuration
    if "dynamics" in cfg.env.env:
        env_config["dynamics_kwargs"] = {
            "action_type": cfg.env.env.dynamics.action_type,
            "dt": cfg.env.env.dynamics.dt,
            "ctrl_dt": cfg.env.env.dynamics.ctrl_dt,
        }
    
    # Add scene configuration
    if "scene" in cfg.env.env:
        env_config["scene_kwargs"] = {
            "path": cfg.env.env.scene.path
        }
    
    return env_config


def create_llm_config(cfg: DictConfig) -> dict:
    """Create LLM configuration from Hydra config"""
    
    return {
        "model": cfg.llm.llm.model,
        "api_key": cfg.llm.llm.api_key,
        "base_url": cfg.llm.llm.base_url,
        "temperature": cfg.llm.llm.temperature,
        "max_tokens": cfg.llm.llm.max_tokens,
        "timeout": cfg.llm.llm.timeout,
        "max_retries": cfg.llm.llm.max_retries,
    }


def create_optimization_config(cfg: DictConfig) -> dict:
    """Create optimization configuration from Hydra config"""
    
    return {
        "iterations": cfg.optimization.iterations,
        "samples": cfg.optimization.samples,
        "training_steps": cfg.task.task.training.steps_per_evaluation,
        "algorithm": cfg.optimization.algorithm,
        "evaluation_episodes": cfg.task.task.training.evaluation_episodes,
        "success_threshold": cfg.task.task.training.success_threshold,
    }


def create_logging_config(cfg: DictConfig) -> dict:
    """Create logging configuration from Hydra config"""
    
    return {
        "level": cfg.logging.level,
        "save_tensorboard": cfg.logging.save_tensorboard,
        "save_all_functions": cfg.logging.save_all_functions,
        "create_visualizations": cfg.logging.create_visualizations,
        "detailed_metrics": cfg.logging.detailed_metrics,
    }


def print_configuration_summary(cfg: DictConfig, logger: logging.Logger):
    """Print a summary of the current configuration"""
    
    logger.info("=" * 60)
    logger.info("VISFLY-EUREKA OPTIMIZATION")
    logger.info("=" * 60)
    logger.info(f"Pipeline: {cfg.pipeline.name}")
    logger.info(f"Task: {cfg.task.task.name} ({cfg.task.task.category})")
    logger.info(f"Environment: {cfg.env.env.name}")
    logger.info(f"LLM: {cfg.llm.llm.provider}/{cfg.llm.llm.model}")
    logger.info(f"Algorithm: {cfg.optimization.algorithm.upper()}")
    logger.info(f"Device: {cfg.execution.device.upper()}")
    logger.info(f"Optimization: {cfg.optimization.iterations} iterations × {cfg.optimization.samples} samples")
    logger.info(f"Output: {cfg.pipeline.output_dir}")
    logger.info("=" * 60)


def run_optimization(cfg: DictConfig, logger: logging.Logger) -> OptimizationReport:
    """Run the complete optimization pipeline"""
    
    # Create configurations
    env_config = create_environment_config(cfg)
    llm_config = create_llm_config(cfg)
    opt_config = create_optimization_config(cfg)
    log_config = create_logging_config(cfg)
    
    # Use Hydra's working directory for outputs - check if we're in a Hydra run directory
    hydra_output_dir = os.getcwd()
    
    # Check if Hydra changed our working directory to a run-specific directory
    import hydra
    from hydra.core.hydra_config import HydraConfig
    
    if HydraConfig.initialized():
        # Get the output directory from Hydra
        hydra_cfg = HydraConfig.get()
        hydra_output_dir = hydra_cfg.runtime.output_dir
        logger.info(f"Using Hydra output directory: {hydra_output_dir}")
    else:
        logger.info(f"Hydra not initialized, using current directory: {hydra_output_dir}")
    
    # Initialize pipeline
    pipeline = EurekaNavigationPipeline(
        task_description=cfg.task.task.description,
        llm_config=llm_config,
        env_config=env_config,
        optimization_config=opt_config,
        logging_config=log_config,
        output_dir=hydra_output_dir
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
    success_pct = (results.successful_samples/results.total_samples*100) if results.total_samples > 0 else 0.0
    logger.info(f"Successful Samples: {results.successful_samples} ({success_pct:.1f}%)")
    
    if results.best_performance:
        logger.info(f"Best Success Rate: {results.best_performance.get('success_rate', 0):.3f}")
        logger.info(f"Best Episode Length: {results.best_performance.get('episode_length', 0):.1f}")
        logger.info(f"Best Score: {results.best_performance.get('score', 0):.4f}")
    
    if results.improvement_metrics:
        improvement = results.improvement_metrics.get('success_rate_improvement', 0)
        relative_improvement = results.improvement_metrics.get('relative_improvement', 0) * 100
        logger.info(f"Success Rate Improvement: {improvement:+.3f} ({relative_improvement:+.1f}%)")
    
    logger.info(f"Execution Time: {results.execution_time:.1f}s")
    logger.info("=" * 60)
    
    # Iteration progression
    if results.iteration_history:
        logger.info("Iteration Progression:")
        for iter_data in results.iteration_history:
            logger.info(f"  Iter {iter_data['iteration']}: "
                       f"Success={iter_data['best_success_rate']:.3f}, "
                       f"Exec Rate={iter_data['execution_rate']:.1%}")


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
        # Print configuration
        logger.info(f"Working directory: {os.getcwd()}")
        logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg, resolve=True)}")
        
        # Validate configuration
        validate_config(cfg, logger)
        
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
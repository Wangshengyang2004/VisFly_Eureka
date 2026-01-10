"""
Bootstrap helpers for composing the Eureka pipeline from Hydra config.

Keep `main.py` thin by moving configuration plumbing and object construction
here (composition root helpers).
"""

from __future__ import annotations

import importlib
import logging
import os
from pathlib import Path
from typing import Any

import torch
import yaml
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from .core.models import OptimizationConfig
from .eureka_visfly import EurekaVisFly
from .modes.base import OptimizationMode
from .modes.eureka_mode import EurekaMode
from .modes.temptuner_mode import TempTunerMode


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_environment_class(env_name: str):
    """
    Load environment class by name.
    
    This function ensures proper package initialization before importing
    environment classes to avoid relative import issues.
    """
    import sys
    from pathlib import Path
    import importlib.util
    
    # Ensure project root is in sys.path for proper imports
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    # Initialize VisFly package first to avoid relative import issues
    # This is needed because envs/*.py files import from VisFly, which uses
    # relative imports that require the package to be properly initialized
    try:
        # Import VisFly as a package to initialize its structure
        import VisFly  # noqa: F401
        # Import key submodules to ensure they're initialized
        import VisFly.utils  # noqa: F401
        import VisFly.envs.base  # noqa: F401
    except ImportError:
        # VisFly might not be available or already imported
        pass
    
    env_registry = {
        "flip": ("envs.FlipEnv", "FlipEnv"),
        "hover": ("envs.HoverEnv", "HoverEnv"),  # Use envs/ not VisFly/envs/ (VisFly version has get_success always returning False)
        "navigation": ("envs.NavigationEnv", "NavigationEnv"),  # Use envs/ not VisFly/envs/
        "racing": ("VisFly.envs.RacingEnv", "RacingEnv"),
        "tracking": ("VisFly.envs.ObjectTrackingEnv", "ObjectTrackingEnv"),
    }

    if env_name not in env_registry:
        raise ValueError(f"Unknown environment: {env_name}")

    module_name, class_name = env_registry[env_name]
    
    # For envs.* modules, use file-based loading to avoid import conflicts
    if module_name.startswith("envs."):
        try:
            # Load directly from file to avoid Python's module search finding VisFly.envs.*
            # module_name is like "envs.NavigationEnv", convert to "envs/NavigationEnv.py"
            env_file = project_root / f"{module_name.replace('.', '/')}.py"
            if not env_file.exists():
                raise FileNotFoundError(f"Environment file not found: {env_file}")
            
            spec = importlib.util.spec_from_file_location(module_name, env_file)
            if spec is None or spec.loader is None:
                raise ImportError(f"Failed to create spec for {module_name}")
            
            module = importlib.util.module_from_spec(spec)
            # Add to sys.modules to make it importable by other code
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            env_class = getattr(module, class_name)
            return env_class
        except Exception as e:
            # Fallback to normal import if file-based loading fails
            try:
                module = importlib.import_module(module_name)
                env_class = getattr(module, class_name)
                return env_class
            except (ImportError, AttributeError) as import_error:
                raise ValueError(
                    f"Failed to import {env_name} environment from {module_name}.{class_name}: {import_error}\n"
                    f"Original error: {e}\n"
                    f"Make sure the environment class exists and all dependencies are properly installed."
                ) from import_error
    else:
        # For VisFly.envs.* modules, use normal import
        try:
            module = importlib.import_module(module_name)
            env_class = getattr(module, class_name)
            return env_class
        except (ImportError, AttributeError) as e:
            raise ValueError(
                f"Failed to import {env_name} environment from {module_name}.{class_name}: {e}\n"
                f"Make sure the environment class exists and all dependencies are properly installed."
            ) from e


def resolve_env_config(cfg: DictConfig, env_name: str) -> DictConfig:
    """Load environment config from file."""
    env_config_path = PROJECT_ROOT / "configs" / "envs" / f"{env_name}.yaml"
    if not env_config_path.exists():
        raise ValueError(
            f"Environment config not found for '{env_name}' at {env_config_path}"
        )
    return OmegaConf.load(env_config_path)


def resolve_task_description(cfg: DictConfig) -> str:
    """Get task description from config. Fail fast if not found."""
    task_description = OmegaConf.select(cfg, "task.task.description")
    if not task_description:
        raise ValueError(
            f"Task description not found in cfg.task.task.description. "
            f"Please ensure your task config includes a description."
        )
    return str(task_description)


def _resolve_path_fields(container: Any) -> Any:
    """Recursively convert relative `path` fields to absolute paths."""

    def _should_convert(key: str, value: Any) -> bool:
        if key is None:
            return False
        if not isinstance(value, str):
            return False
        if value == "":
            return False
        lowered = value.lower()
        if lowered.startswith(("s3://", "http://", "https://")):
            return False
        if not ("/" in value or value.startswith(".")):
            return False
        return key == "path" or key.endswith("_path") or key.endswith("_dir")

    def _make_absolute(path_str: str) -> str:
        if os.path.isabs(path_str):
            return path_str
        try:
            return to_absolute_path(path_str)
        except Exception:
            return str((PROJECT_ROOT / path_str).resolve())

    if isinstance(container, dict):
        resolved: dict[str, Any] = {}
        for key, value in container.items():
            resolved_value = _resolve_path_fields(value)
            if _should_convert(str(key), value):
                resolved_value = _make_absolute(value)
            resolved[key] = resolved_value
        return resolved

    if isinstance(container, list):
        return [_resolve_path_fields(item) for item in container]

    return container


def prepare_env_config(env_config: DictConfig) -> dict:
    """Convert environment config to dict, resolving relative paths."""
    env_kwargs = OmegaConf.to_container(env_config, resolve=True)
    env_kwargs = _resolve_path_fields(env_kwargs)

    if "target" in env_kwargs:
        env_kwargs["target"] = torch.tensor([env_kwargs["target"]])

    return env_kwargs


def load_llm_config(cfg: DictConfig, opt_config: OptimizationConfig = None) -> dict:
    """Load and prepare LLM configuration with API keys."""
    api_keys_path = PROJECT_ROOT / "configs" / "api_keys.yaml"
    with open(api_keys_path, "r") as f:
        api_config = yaml.safe_load(f)
        providers = api_config["providers"]
        vendor = str(cfg.llm.llm.vendor)
        if vendor == "default":
            vendor = providers["default"]
        vendor_config = providers[vendor]

    llm_config = {
        "model": cfg.llm.llm.model,
        "api_key": vendor_config["api_key"],
        "base_url": vendor_config["base_url"],
        "temperature": cfg.llm.llm.temperature,
        "timeout": cfg.llm.llm.timeout,
        "max_retries": cfg.llm.llm.max_retries,
    }

    # max_tokens is optional - if not specified, API will use default (unlimited)
    # Only add it if it exists in the config
    if hasattr(cfg.llm.llm, "max_tokens") and cfg.llm.llm.max_tokens is not None:
        llm_config["max_tokens"] = cfg.llm.llm.max_tokens

    if hasattr(cfg.llm.llm, "thinking") and hasattr(cfg.llm.llm.thinking, "enabled"):
        llm_config["thinking_enabled"] = cfg.llm.llm.thinking.enabled

    if hasattr(cfg.llm.llm, "batching"):
        batching = cfg.llm.llm.batching
        if hasattr(batching, "strategy"):
            llm_config["batching_strategy"] = batching.strategy
        if hasattr(batching, "supports_n_parameter"):
            llm_config["supports_n_parameter"] = batching.supports_n_parameter
        if hasattr(batching, "max_concurrent"):
            llm_config["max_concurrent"] = batching.max_concurrent

    # Load prompt config from main config (moved from individual LLM configs)
    if "prompt" in cfg:
        prompt_cfg = OmegaConf.to_container(cfg.prompt, resolve=True)
        include_api_doc = prompt_cfg.get("include_api_doc")
        if include_api_doc is not None:
            llm_config["include_api_doc"] = bool(include_api_doc)
        api_doc_path = prompt_cfg.get("api_doc_path")
        if api_doc_path:
            llm_config["api_doc_path"] = api_doc_path
        include_human_reward = prompt_cfg.get("include_human_reward")
        if include_human_reward is not None:
            llm_config["include_human_reward"] = bool(include_human_reward)

    # Add history_window_size from optimization config
    if opt_config is not None and hasattr(opt_config, "history_window_size"):
        llm_config["history_window_size"] = opt_config.history_window_size

    return llm_config


def create_optimization_config(cfg: DictConfig) -> OptimizationConfig:
    """Create optimization configuration from config."""
    record_video = bool(getattr(cfg.optimization, "record_video", False))
    return OptimizationConfig(
        iterations=cfg.optimization.iterations,
        samples=cfg.optimization.samples,
        algorithm=cfg.optimization.algorithm,
        evaluation_episodes=cfg.optimization.evaluation_episodes,
        record_video=record_video,
        gpu_memory_requirement_mb=getattr(
            cfg.optimization, "gpu_memory_requirement_mb", 2048
        ),
        history_window_size=getattr(
            cfg.optimization, "history_window_size", 2
        ),
    )


def prepare_eval_env_config(env_config: DictConfig, base_env_config: dict) -> dict:
    """Prepare evaluation environment configuration."""
    if "eval_env" in env_config:
        return prepare_env_config(env_config.eval_env)
    return base_env_config.copy()


def create_eureka_controller(cfg: DictConfig, logger: logging.Logger) -> EurekaVisFly:
    """Create and configure EurekaVisFly controller from config."""
    # Get environment name from task config
    env_name = OmegaConf.select(cfg, "task.task.envs")
    if not env_name:
        raise ValueError(
            f"Environment name not found in cfg.task.task.envs. "
            f"Please ensure your task config includes an 'envs' field."
        )
    env_name = str(env_name)
    logger.info(f"Using environment from task config: {env_name}")

    # Resolve task description
    task_description = resolve_task_description(cfg)

    env_class = load_environment_class(env_name)

    env_config = resolve_env_config(cfg, env_name)
    env_kwargs = prepare_env_config(env_config.env)

    # Create opt_config before llm_config so we can pass history_window_size
    opt_config = create_optimization_config(cfg)
    llm_config = load_llm_config(cfg, opt_config)

    eval_env_config = prepare_eval_env_config(env_config, env_kwargs)

    execution_device = str(getattr(cfg.execution, "device", "cpu")).lower()
    workload_ratio = getattr(cfg.execution, "workload_ratio", 2)

    if execution_device == "cuda" and torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        max_workers = max(1, gpu_count * workload_ratio)
        logger.info(
            "Detected %d GPU(s), using workload_ratio=%d, max_workers=%d",
            gpu_count,
            workload_ratio,
            max_workers,
        )
    else:
        max_workers = max(1, workload_ratio)
        logger.info(
            "CPU mode, using workload_ratio=%d, max_workers=%d",
            workload_ratio,
            max_workers,
        )

    # Check if coefficient tuning mode is enabled
    use_coefficient_tuning = bool(getattr(cfg.optimization, "use_coefficient_tuning", False))

    return EurekaVisFly(
        env_class=env_class,
        task_description=task_description,
        llm_config=llm_config,
        env_kwargs=env_kwargs,
        optimization_config=opt_config,
        device=execution_device,
        max_workers=max_workers,
        eval_env_config=eval_env_config,
        use_coefficient_tuning=use_coefficient_tuning,
    )


def create_mode(
    cfg: DictConfig, logger: logging.Logger, output_dir: str | Path
) -> OptimizationMode:
    """
    Create an optimization mode from Hydra config.

    This is the single entrypoint `main.py` should use when selecting between
    different optimization workflows (eureka, temptuner, ...).
    """
    mode_name = getattr(getattr(cfg, "mode", None), "name", "eureka")
    mode_name = str(mode_name).lower()

    # All modes currently share the same EurekaVisFly controller; they differ
    # only in configuration (Hydra groups) and how the results are interpreted.
    if mode_name == "eureka":
        controller = create_eureka_controller(cfg, logger)
        return EurekaMode(controller=controller, output_dir=output_dir)

    if mode_name == "temptuner":
        # Enable coefficient tuning for temptuner mode
        if not hasattr(cfg.optimization, "use_coefficient_tuning"):
            from omegaconf import OmegaConf
            OmegaConf.set_struct(cfg, False)
            cfg.optimization.use_coefficient_tuning = True
        controller = create_eureka_controller(cfg, logger)
        from .modes.temptuner_mode import TempTunerMode
        return TempTunerMode(controller=controller, output_dir=output_dir)

    raise ValueError(f"Unknown mode: {mode_name}")


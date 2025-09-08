"""
Standalone evaluation worker invoked by SubprocessRewardEvaluator.

This script runs in a separate Python process to train and evaluate a model
with an injected reward function, returning JSON results via an output file.

Key behaviors:
- Adds project_root and project_root/VisFly to sys.path (from config)
- Dynamically imports the specified environment class
- Injects the provided reward function (expects a get_reward(self) function)
- Supports BPTT algorithm for training (extensible)
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict

import yaml


def _setup_paths(project_root: Path) -> None:
    """Ensure project directories are on sys.path without hardcoding absolute paths."""
    root = str(project_root.resolve())
    visfly = str((project_root / "VisFly").resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    if visfly not in sys.path:
        sys.path.insert(0, visfly)


def _import_env_class(env_class_path: str):
    """Import environment class from its full dotted path."""
    module_path, _, class_name = env_class_path.rpartition(".")
    if not module_path:
        raise ImportError(f"Invalid env_class path: {env_class_path}")
    module = importlib.import_module(module_path)
    env_class = getattr(module, class_name)
    return env_class


def _inject_reward_to_class(env_class: Any, reward_code: str, logger: logging.Logger) -> bool:
    """Inject user-provided get_reward into environment class (like run.py)."""
    import numpy as np  # noqa: F401
    import torch as th
    # Use same execution context as run.py
    exec_globals: Dict[str, Any] = {
        "torch": th,
        "th": th,
        "np": __import__("numpy"),
        "numpy": __import__("numpy"),
    }
    try:
        exec(reward_code, exec_globals)
    except Exception as e:
        logger.error(f"Executing reward code failed: {e}")
        return False

    get_reward = exec_globals.get("get_reward")
    if get_reward is None:
        logger.error("No get_reward function found in reward code")
        return False

    try:
        # Class-level injection like run.py does
        env_class.get_reward = get_reward
        logger.debug("Successfully injected reward function into class")
        return True
    except Exception as e:
        logger.error(f"Binding reward function to class failed: {e}")
        return False


# Removed legacy _inject_reward function - now using class-level injection


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def evaluate_main(config_path: Path, output_path: Path) -> int:
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    logger = logging.getLogger(__name__)

    try:
        with open(config_path, "r") as f:
            cfg = json.load(f)

        reward_code: str = cfg["reward_code"]
        identifier: str = cfg["identifier"]
        env_config: Dict[str, Any] = cfg.get("env_config", {})
        eval_env_config: Dict[str, Any] = cfg.get("eval_env_config", env_config)
        optimization_config: Dict[str, Any] = cfg.get("optimization_config", {})
        env_class_path: str = cfg["env_class"]
        output_dir = Path(cfg.get("output_dir", str(output_path.parent)))
        project_root = Path(cfg.get("project_root", Path(__file__).resolve().parents[2]))

        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

        # Setup PYTHONPATH dynamically
        _setup_paths(project_root)
        logger.info(f"Starting evaluation for {identifier}")

        # Resolve environment class
        logger.info("[stage] importing environment class ...")
        EnvClass = _import_env_class(env_class_path)
        logger.info("[stage] environment class imported")

        # Inject reward into CLASS first (like run.py)
        logger.info("[stage] injecting reward into environment class ...")
        if not _inject_reward_to_class(EnvClass, reward_code, logger):
            result = {
                "success": False,
                "error": "Failed to inject reward function into class",
                "identifier": identifier,
            }
            with open(output_path, "w") as f:
                json.dump(result, f, indent=2)
            return 0
        logger.info("[stage] reward injected into environment class")

        # Algorithm and device setup
        algorithm = optimization_config["algorithm"].lower()
        eval_episodes = int(optimization_config["evaluation_episodes"])

        # Load algorithm config - extract environment name from class path
        logger.info("[stage] loading algorithm config ...")
        # Extract env name from class path like "envs.NavigationEnv.NavigationEnv" -> "navigation"
        env_name = env_class_path.split('.')[-2].lower().replace('env', '')  # NavigationEnv -> navigation
        alg_cfg_path = project_root / "configs" / "algs" / env_name / f"{algorithm}.yaml"
        
        alg_cfg = _load_yaml(alg_cfg_path)
        alg_params = alg_cfg["algorithm"].copy()
        learn_params = alg_cfg["learn"]
        
        # Use configured training steps (respect the config!)
        train_steps = int(learn_params["total_timesteps"])
        log_interval = learn_params["log_interval"]
        
        logger.info(f"Training configuration: {train_steps} timesteps, log_interval={log_interval}")
        
        # Keep devices separate - env and algorithm use their configured devices
        env_device = env_config["device"]
        alg_device = alg_params["device"]
        logger.info(f"Environment device: {env_device}, Algorithm device: {alg_device}")
        
        # Set requires_grad for BPTT
        env_config = {**env_config, "requires_grad": True}

        # Create training environment (will inherit injected reward function)
        logger.info("[stage] constructing training env ...")
        env = EnvClass(**env_config)
        logger.info("[stage] training env constructed with injected reward")

        # Note: Do NOT call env.reset() here - let the algorithm handle it properly
        

        # Create BPTT algorithm
        logger.info("[stage] instantiating BPTT algorithm ...")
        from algorithms.BPTT import BPTT

        alg_params["env"] = env
        alg_params.setdefault("policy", "SimplePolicy")
        
        # Add TensorBoard logging support
        tensorboard_dir = Path(output_dir) / "tensorboard"
        tensorboard_dir.mkdir(exist_ok=True)
        alg_params["tensorboard_log"] = str(tensorboard_dir)
        logger.info(f"TensorBoard logs will be saved to: {tensorboard_dir}")
        
        model = BPTT(**alg_params)
        logger.info("[stage] BPTT algorithm instantiated")

        # Train
        logger.info(f"[stage] starting training for {train_steps} steps ...")
        t0 = time.time()
        
        # Prepare training arguments
        train_args = {"total_timesteps": int(train_steps)}
        if log_interval is not None:
            train_args["log_interval"] = log_interval
        
        # Use getattr to avoid static type check issues
        getattr(model, "learn")(**train_args)
        training_time = time.time() - t0
        logger.info(f"Training completed in {training_time:.2f}s")

        # Create evaluation env (requires_grad False)
        eval_env_config = {**eval_env_config, "requires_grad": False}
        logger.info("[stage] constructing eval env ...")
        env_eval = EnvClass(**eval_env_config)
        logger.info("[stage] eval env constructed (inherits injected reward from class)")

        # Run evaluation episodes
        import numpy as np
        import torch as th

        success_count = 0
        episode_lengths = []
        final_rewards = []
        # Use environment's configured max_episode_steps
        max_steps = int(eval_env_config["max_episode_steps"])

        for _ in range(eval_episodes):
            obs = env_eval.reset()
            done_flag = False
            step_count = 0
            episode_reward = 0.0

            while not done_flag and step_count < max_steps:
                if hasattr(model, "predict"):
                    action = model.predict(obs)
                    if isinstance(action, tuple):
                        action = action[0]
                else:
                    action = env_eval.action_space.sample()

                obs, reward, done_arr, info = env_eval.step(action)

                if isinstance(reward, th.Tensor):
                    episode_reward += float(reward.mean().item())
                elif isinstance(reward, np.ndarray):
                    episode_reward += float(reward.mean())
                else:
                    episode_reward += float(reward)

                if isinstance(done_arr, (np.ndarray, list)):
                    done_flag = bool(np.any(done_arr))
                elif isinstance(done_arr, th.Tensor):
                    done_flag = bool(done_arr.any().item())
                else:
                    done_flag = bool(done_arr)

                step_count += 1

            if hasattr(env_eval, "get_success"):
                success_tensor = env_eval.get_success()
                if hasattr(success_tensor, "any"):
                    try:
                        success = bool(success_tensor.any().item())
                    except Exception:
                        success = bool(success_tensor.any())
                else:
                    success = bool(success_tensor)
            else:
                success = episode_reward > 0

            if success:
                success_count += 1

            episode_lengths.append(step_count)
            final_rewards.append(episode_reward)

        success_rate = success_count / max(1, eval_episodes)
        avg_len = sum(episode_lengths) / max(1, len(episode_lengths))
        avg_final_reward = sum(final_rewards) / max(1, len(final_rewards))

        result = {
            "success": True,
            "identifier": identifier,
            "reward_code": reward_code,
            "success_rate": success_rate,
            "episode_length": avg_len,
            "training_time": training_time,
            "final_reward": avg_final_reward,
            "convergence_step": train_steps,
            "output_dir": str(output_dir),
        }

        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        return 0

    except Exception as e:
        tb = traceback.format_exc()
        try:
            with open(output_path, "w") as f:
                json.dump(
                    {
                        "success": False,
                        "error": str(e),
                        "traceback": tb,
                        "identifier": cfg.get("identifier", "unknown") if "cfg" in locals() else "unknown",
                    },
                    f,
                    indent=2,
                )
        except Exception:
            # If even writing fails, log to stderr
            logging.error(tb)
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Eureka evaluation worker")
    parser.add_argument("config_file", type=str)
    parser.add_argument("output_file", type=str)
    args = parser.parse_args()

    config_path = Path(args.config_file).resolve()
    output_path = Path(args.output_file).resolve()
    return evaluate_main(config_path, output_path)


if __name__ == "__main__":
    raise SystemExit(main())

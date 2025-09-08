#!/usr/bin/env python3
"""
Simplified VisFly Training Wrapper for Eureka Pipeline

This wrapper uses YAML config files for algorithm and environment settings,
following the obj_track project pattern. Supports BPTT, PPO, and SHAC algorithms
with clean configuration management.
"""

import sys
import os
import time
import logging
import torch
import torch as th
import numpy as np
import argparse
import yaml
from pathlib import Path

# Add VisFly to path dynamically based on current file location
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent  # Go up two levels to project root
visfly_path = project_root / "VisFly"

sys.path.append(str(visfly_path))
sys.path.append(str(project_root))

# Import VisFly components
from VisFly.utils.algorithms.BPTT import BPTT
from VisFly.utils.algorithms.PPO import PPO
from VisFly.utils.algorithms.shac import TemporalDifferBase as SHAC
from VisFly.envs.NavigationEnv import NavigationEnv

# Module logger
logger = logging.getLogger(__name__)

# Disable gradient anomaly detection for physics-based algorithms
torch.autograd.set_detect_anomaly(False)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="VisFly Training Wrapper with Config Files"
    )
    parser.add_argument(
        "--env", "-e", type=str, default="navigation", help="Environment name"
    )
    parser.add_argument(
        "--algorithm",
        "-a",
        type=str,
        default="bptt",
        choices=["bptt", "ppo", "shac"],
        help="Algorithm name",
    )
    parser.add_argument(
        "--train", "-t", type=int, default=1, help="Training mode (1) or test mode (0)"
    )
    parser.add_argument(
        "--comment", "-c", type=str, default=None, help="Experiment comment"
    )
    parser.add_argument(
        "--seed", "-s", type=int, default=None, help="Random seed override"
    )
    parser.add_argument(
        "--reward_function_path",
        type=str,
        default=None,
        help="Path to custom reward function",
    )
    parser.add_argument(
        "--save_dir", type=str, default=None, help="Directory to save results"
    )

    # Config overrides
    parser.add_argument(
        "--learning_steps", type=int, default=None, help="Override learning steps"
    )
    parser.add_argument(
        "--num_agents", type=int, default=None, help="Override number of agents"
    )
    parser.add_argument("--device", type=str, default=None, help="Override device")

    return parser.parse_args()


def load_config_files(env_name: str, algorithm: str):
    """Load environment and algorithm config files"""
    config_root = Path(__file__).parent.parent.parent / "configs"

    # Load environment config
    env_config_path = config_root / "envs" / f"{env_name}.yaml"
    if not env_config_path.exists():
        raise FileNotFoundError(f"Environment config not found: {env_config_path}")

    with open(env_config_path, "r") as f:
        env_config = yaml.safe_load(f)

    # Load algorithm config
    alg_config_path = config_root / "algs" / env_name / f"{algorithm}.yaml"
    if not alg_config_path.exists():
        raise FileNotFoundError(f"Algorithm config not found: {alg_config_path}")

    with open(alg_config_path, "r") as f:
        alg_config = yaml.safe_load(f)

    logger.info("Loaded configs")
    logger.info("  Environment: %s", env_config_path)
    logger.info("  Algorithm: %s", alg_config_path)

    return env_config, alg_config


def apply_config_overrides(env_config: dict, alg_config: dict, args):
    """Apply command line overrides to configs"""

    # Apply algorithm environment overrides
    if "env_overrides" in alg_config:
        env_overrides = alg_config["env_overrides"]
        for key, value in env_overrides.items():
            env_config["env"][key] = value
            logger.info("Applied env override: %s = %s", key, value)

    # Apply command line overrides
    if args.learning_steps is not None:
        alg_config["learn"]["total_timesteps"] = args.learning_steps
        logger.info("Override learning_steps: %s", args.learning_steps)

    if args.num_agents is not None:
        env_config["env"]["num_agent_per_scene"] = args.num_agents
        logger.info("Override num_agents: %s", args.num_agents)

    if args.device is not None:
        env_config["env"]["device"] = args.device
        alg_config["algorithm"]["device"] = args.device
        logger.info("Override device: %s", args.device)

    if args.seed is not None:
        alg_config["algorithm"]["seed"] = args.seed
        logger.info("Override seed: %s", args.seed)

    if args.comment is not None:
        alg_config["comment"] = args.comment
        logger.info("Set comment: %s", args.comment)

    return env_config, alg_config


def inject_reward_function(reward_function_code: str, env_class):
    """Inject custom reward function into the environment class"""
    if reward_function_code:
        try:
            exec_globals = {"torch": torch, "th": th, "np": np, "numpy": np}

            exec(reward_function_code, exec_globals)
            env_class.get_reward = exec_globals["get_reward"]

            logger.info("Successfully injected custom reward function")

        except Exception as e:
            logger.exception("Failed to inject reward function: %s", e)
            raise


def create_environment(env_config: dict, env_name: str):
    """Create environment from config"""
    logger.info("Creating %s environment...", env_name)

    # Convert target to tensor if it's a list
    if "target" in env_config["env"] and isinstance(env_config["env"]["target"], list):
        env_config["env"]["target"] = torch.tensor([env_config["env"]["target"]])

    # Create environment based on name
    if env_name == "navigation":
        env = NavigationEnv(**env_config["env"])
    else:
        raise ValueError(f"Unsupported environment: {env_name}")

    logger.info(
        "Environment created with %s agents",
        env_config['env']['num_agent_per_scene'],
    )
    return env


def create_algorithm(algorithm: str, alg_config: dict, env, save_folder: str):
    """Create algorithm from config"""

    logger.info("Creating %s algorithm...", algorithm.upper())

    algorithm_params = alg_config["algorithm"].copy()
    comment = alg_config.get("comment", f"{algorithm}_experiment")

    if algorithm == "bptt":
        return BPTT(env=env, comment=comment, save_path=save_folder, **algorithm_params)

    elif algorithm == "ppo":
        # Ensure environment tensor output is disabled for PPO
        try:
            env.tensor_output = False
        except AttributeError:
            pass

        # Ensure tensorboard log directory exists
        import os
        tensorboard_dir = os.path.join(save_folder, "tensorboard")
        os.makedirs(tensorboard_dir, exist_ok=True)
        
        logger.info(f"TensorBoard logs will be saved to: {tensorboard_dir}")
        
        return PPO(
            env=env, comment=comment, tensorboard_log=tensorboard_dir, **algorithm_params
        )

    elif algorithm == "shac":
        return SHAC(env=env, comment=comment, save_path=save_folder, **algorithm_params)

    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")


def validate_critical_config(env_config: dict, alg_config: dict) -> None:
    """Minimal validation for essential fields; raises ValueError on issues."""
    missing = []
    # env config
    if not isinstance(env_config, dict) or "env" not in env_config:
        raise ValueError("env_config must contain 'env' section")
    env_required = ["num_agent_per_scene", "device"]
    for k in env_required:
        if k not in env_config["env"]:
            missing.append(f"env.{k}")

    # algorithm config
    if not isinstance(alg_config, dict):
        raise ValueError("alg_config must be a dict")
    if "learn" not in alg_config or "total_timesteps" not in alg_config["learn"]:
        missing.append("learn.total_timesteps")
    if "algorithm" not in alg_config:
        missing.append("algorithm section")

    if missing:
        raise ValueError("Missing required config keys: " + ", ".join(missing))


def run_training(args):
    """Run the training with config-based setup"""

    logger.info("Starting training: %s with %s", args.env, args.algorithm.upper())

    try:
        # Load config files
        env_config, alg_config = load_config_files(args.env, args.algorithm)

        # Apply overrides
        env_config, alg_config = apply_config_overrides(env_config, alg_config, args)

        # Validate configs
        validate_critical_config(env_config, alg_config)

        # Setup save directory
        if args.save_dir:
            save_folder = args.save_dir
        else:
            save_folder = os.path.dirname(os.path.abspath(__file__)) + "/saved/"
        os.makedirs(save_folder, exist_ok=True)

        # Create environment
        env = create_environment(env_config, args.env)

        # Reset environment to initialize
        logger.info("Resetting environment...")
        env.reset()

        # Load custom reward function if provided
        if args.reward_function_path and os.path.exists(args.reward_function_path):
            with open(args.reward_function_path, "r") as f:
                reward_code = f.read()
            inject_reward_function(reward_code, NavigationEnv)

        # Create algorithm
        model = create_algorithm(args.algorithm, alg_config, env, save_folder)

        if args.train:
            # Train the model
            learning_params = alg_config["learn"]
            logger.info(
                "Starting training for %s steps...",
                learning_params['total_timesteps'],
            )

            start_time = time.time()

            if args.algorithm in ["bptt", "shac"]:
                model.learn(int(learning_params["total_timesteps"]))
            elif args.algorithm == "ppo":
                model.learn(total_timesteps=int(learning_params["total_timesteps"]))

            model.save()

            training_time = time.time() - start_time
            logger.info("Training completed in %.2f seconds", training_time)

            # Save training info
            import json

            info_file = os.path.join(save_folder, "training_info.json")
            training_info = {
                "environment": args.env,
                "algorithm": args.algorithm,
                "training_time": training_time,
                "total_timesteps": learning_params["total_timesteps"],
                "config_files": {
                    "env": f"configs/envs/{args.env}.yaml",
                    "algorithm": f"configs/algs/{args.env}/{args.algorithm}.yaml",
                },
            }

            with open(info_file, "w") as f:
                json.dump(training_info, f, indent=2)

            logger.info("Saved training info to %s", info_file)

        else:
            logger.info("Test mode - skipping training")

        return True

    except Exception as e:
        logger.exception("Training failed: %s", e)
        return False


def main():
    """Main entry point"""
    # Configure root logger if not configured by parent
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    logger.info("VisFly Training Wrapper v2 (Config-Based)")

    args = parse_args()

    logger.info("Configuration:")
    logger.info("  Environment: %s", args.env)
    logger.info("  Algorithm: %s", args.algorithm)
    logger.info("  Mode: %s", 'Training' if args.train else 'Testing')

    try:
        success = run_training(args)
        if success:
            logger.info("Operation completed successfully")
            return 0
        else:
            logger.error("Operation failed")
            return 1

    except Exception as e:
        logger.exception("Wrapper failed: %s", e)
        return 1


if __name__ == "__main__":
    exit(main())

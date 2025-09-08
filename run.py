#!/usr/bin/env python3
"""
Unified Run System for VisFly-Eureka

This is the main entry point for training and testing environments with
algorithm configs, inspired by obj_track's run.py but integrated with
the Eureka pipeline.
"""

import sys
import os
import json
import logging
import torch as th
import traceback
import argparse
from pathlib import Path
from typing import Any, cast

# Set matplotlib backend for headless servers
import matplotlib

matplotlib.use("Agg")

# Add paths (project root first, then VisFly)
sys.path.insert(0, str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "VisFly"))

# Import environments (use absolute paths to avoid VisFly conflicts)
import importlib.util
from pathlib import Path

# Module logger
logger = logging.getLogger(__name__)


def load_env_class(env_name, class_name):
    """Dynamically load environment class from specific file"""
    env_file = Path(__file__).parent / "envs" / f"{env_name}.py"
    if not env_file.exists():
        raise FileNotFoundError(f"Env file not found: {env_file}")
    spec = importlib.util.spec_from_file_location(class_name, str(env_file))
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load spec for {class_name} from {env_file}")
    module = importlib.util.module_from_spec(spec)
    # spec.loader is not None here
    spec.loader.exec_module(module)
    return getattr(module, class_name)


# Load environment classes dynamically to avoid import conflicts
NavigationEnv = load_env_class("NavigationEnv", "NavigationEnv")
HoverEnv = load_env_class("HoverEnv", "HoverEnv")
ObjectTrackingEnv = load_env_class("ObjectTrackingEnv", "ObjectTrackingEnv")
CatchEnv = load_env_class("CatchEnv", "CatchEnv")
LandingEnv = load_env_class("LandingEnv", "LandingEnv")
RacingEnv = load_env_class("RacingEnv", "RacingEnv")
TrackingEnv = load_env_class("TrackingEnv", "TrackingEnv")
FlipEnv = load_env_class("FlipEnv", "FlipEnv")
VisLandingEnv = load_env_class("VisLanding", "VisLandingEnv")

# Import VisFly components
from VisFly.utils.common import load_yaml_config
from VisFly.utils.policies import extractors

# Import algorithms (use updated ones from /algorithms)
from algorithms.BPTT import BPTT
from VisFly.utils.algorithms.PPO import PPO
from algorithms.SHAC import SHAC


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run VisFly-Eureka experiments")
    parser.add_argument(
        "--env",
        "-e",
        type=str,
        default="navigation",
        choices=[
            "navigation",
            "hover",
            "object_tracking",
            "tracking",
            "catch",
            "landing",
            "racing",
            "flip",
            "vis_landing",
        ],
        help="Environment to use",
    )
    parser.add_argument(
        "--algorithm",
        "-a",
        type=str,
        default="bptt",
        choices=["bptt", "ppo", "shac"],
        help="Algorithm to use",
    )
    parser.add_argument(
        "--train",
        "-t",
        type=int,
        default=1,
        help="Training mode (1) or testing mode (0)",
    )
    parser.add_argument(
        "--comment", "-c", type=str, default=None, help="Experiment comment"
    )
    parser.add_argument("--seed", "-s", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--weight", "-w", type=str, default=None, help="Model weights file for testing"
    )
    parser.add_argument(
        "--reward_function_path",
        type=str,
        default=None,
        help="Path to custom reward function for injection",
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


def load_configs(env_name: str, algorithm: str):
    """Load environment and algorithm configurations"""
    config_root = Path(__file__).parent / "configs"

    # Load environment config
    env_config_path = config_root / "envs" / f"{env_name}.yaml"
    if not env_config_path.exists():
        raise FileNotFoundError(f"Environment config not found: {env_config_path}")
    env_config = load_yaml_config(str(env_config_path))

    # Load algorithm config
    alg_config_path = config_root / "algs" / env_name / f"{algorithm}.yaml"
    if not alg_config_path.exists():
        raise FileNotFoundError(f"Algorithm config not found: {alg_config_path}")
    alg_config = load_yaml_config(str(alg_config_path))

    # Configs loaded silently

    return env_config, alg_config


def apply_config_overrides(env_config: dict, alg_config: dict, args):
    """Apply command line overrides to configs"""

    # Apply algorithm environment overrides
    if "env_overrides" in alg_config:
        for key, value in alg_config["env_overrides"].items():
            env_config["env"][key] = value

    # Apply command line overrides
    if args.learning_steps is not None:
        alg_config["learn"]["total_timesteps"] = args.learning_steps

    if args.num_agents is not None:
        env_config["env"]["num_agent_per_scene"] = args.num_agents

    if args.device is not None:
        env_config["env"]["device"] = args.device
        alg_config["algorithm"]["device"] = args.device

    if args.comment is not None:
        alg_config["comment"] = args.comment
    else:
        # Generate default comment
        alg_config["comment"] = f"{args.env}_{args.algorithm}"

    return env_config, alg_config


def inject_reward_function(reward_function_path: str, env_class):
    """Inject custom reward function from Eureka into environment class"""
    if reward_function_path and os.path.exists(reward_function_path):
        try:
            with open(reward_function_path, "r") as f:
                reward_code = f.read()

            # Create execution globals
            exec_globals = {
                "torch": th,
                "th": th,
                "np": __import__("numpy"),
                "numpy": __import__("numpy"),
            }

            # Execute the reward function code
            exec(reward_code, exec_globals)

            # Replace the get_reward method in the class
            if "get_reward" in exec_globals:
                env_class.get_reward = exec_globals["get_reward"]
                logger.info(
                    "Successfully injected reward function from %s",
                    reward_function_path,
                )
                return True
            else:
                logger.warning(
                    "No get_reward function found in %s", reward_function_path
                )
                return False

        except Exception as e:
            logger.exception("Failed to inject reward function: %s", e)
            return False

    return False


def create_environment(env_name: str, env_config: dict, is_training: bool = True):
    """Create environment instance"""

    env_classes = get_env_class_registry()

    if env_name not in env_classes:
        raise ValueError(f"Unknown environment: {env_name}")

    env_class = env_classes[env_name]
    config_key = "env" if is_training else "eval_env"

    # Make a copy to avoid modifying the original config
    env_kwargs = env_config[config_key].copy()

    # Remove non-constructor parameters
    env_kwargs.pop("name", None)  # Remove name if present

    # Convert target to tensor if it's a list
    if "target" in env_kwargs and isinstance(env_kwargs["target"], list):
        env_kwargs["target"] = th.tensor([env_kwargs["target"]])

    env = env_class(**env_kwargs)

    return env


def create_algorithm(algorithm: str, alg_config: dict, env, save_folder: str):
    """Create algorithm instance"""

    algorithm_params = alg_config["algorithm"].copy()
    comment = alg_config.get("comment", f"{algorithm}_experiment")

    # Algorithm mapping
    if algorithm == "bptt":
        return BPTT(env=env, comment=comment, save_path=save_folder, **algorithm_params)

    elif algorithm == "ppo":
        # Ensure environment tensor output is disabled for PPO
        try:
            env.tensor_output = False
        except AttributeError:
            pass

        return PPO(
            env=env, comment=comment, tensorboard_log=save_folder, **algorithm_params
        )

    elif algorithm == "shac":
        return SHAC(env=env, comment=comment, save_path=save_folder, **algorithm_params)

    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")


def get_env_class_registry():
    """Get mapping of environment names to classes"""
    return {
        "navigation": NavigationEnv,
        "hover": HoverEnv,
        "object_tracking": ObjectTrackingEnv,
        "tracking": TrackingEnv,
        "catch": CatchEnv,
        "landing": LandingEnv,
        "racing": RacingEnv,
        "flip": FlipEnv,
        "vis_landing": VisLandingEnv,
    }


def setup_save_directory(env_name: str) -> str:
    """Create and return save directory path"""
    save_folder = Path(__file__).parent / "results" / "training" / env_name
    save_folder.mkdir(parents=True, exist_ok=True)
    return str(save_folder) + "/"


def apply_custom_reward(reward_function_path: str, env_name: str, train_env):
    """Apply custom reward function if provided"""
    if reward_function_path:
        env_classes = get_env_class_registry()
        if env_name not in env_classes:
            raise ValueError(f"Unknown environment for reward injection: {env_name}")
        
        inject_reward_function(reward_function_path, env_classes[env_name])
        # Reset environment to apply new reward function
        train_env.reset()


def load_model_weights(model, weight_path: str, save_folder: str, train_env):
    """Load existing model weights if specified"""
    if weight_path:
        full_weight_path = save_folder + weight_path
        model.load(path=full_weight_path, env=train_env)


def execute_training(model, algorithm: str, learning_params: dict):
    """Execute the training process based on algorithm"""
    m_any = cast(Any, model)
    if algorithm in ["bptt", "shac"]:
        m_any.learn(**learning_params)
    elif algorithm == "ppo":
        m_any.learn(total_timesteps=int(learning_params["total_timesteps"]))
    else:
        raise ValueError(f"Unknown training algorithm: {algorithm}")


def run_training(args):
    """Execute training workflow"""
    try:
        logger.info(
            "Starting training: env=%s algorithm=%s comment=%s",
            args.env,
            args.algorithm,
            args.comment,
        )
        
        # Load and prepare configurations
        env_config, alg_config = load_configs(args.env, args.algorithm)
        env_config, alg_config = apply_config_overrides(env_config, alg_config, args)

        # Setup directories and environment
        save_folder = setup_save_directory(args.env)
        train_env = create_environment(args.env, env_config, is_training=True)

        # Apply custom reward function if provided
        apply_custom_reward(args.reward_function_path, args.env, train_env)

        # Create and configure algorithm
        model = create_algorithm(args.algorithm, alg_config, train_env, save_folder)
        load_model_weights(model, args.weight, save_folder, train_env)

        # Execute training
        learning_params = alg_config["learn"]
        execute_training(model, args.algorithm, learning_params)

        # Save results
        model.save()
        logger.info("Training completed and model saved.")
        return True

    except Exception as e:
        logger.exception("Training failed: %s", e)
        return False


def run_testing(args):
    """Execute testing workflow"""

    try:
        logger.info(
            "Starting testing: env=%s algorithm=%s weight=%s",
            args.env,
            args.algorithm,
            args.weight,
        )
        # Load configurations
        env_config, alg_config = load_configs(args.env, args.algorithm)
        env_config, alg_config = apply_config_overrides(env_config, alg_config, args)

        # Setup directories
        save_folder = Path(__file__).parent / "results" / "training" / args.env
        test_folder = Path(__file__).parent / "results" / "testing" / args.env
        test_folder.mkdir(parents=True, exist_ok=True)

        # Create evaluation environment
        test_env = create_environment(args.env, env_config, is_training=False)

        # Load trained model
        if args.weight is None:
            raise ValueError("Weight file required for testing mode")

        weight_path = save_folder / args.weight

        # Create algorithm and load weights
        algorithm_classes = {"bptt": BPTT, "ppo": PPO, "shac": SHAC}
        model = algorithm_classes[args.algorithm].load(str(weight_path), env=test_env)

        # Simple evaluation loop
        test_env.reset()
        total_reward = 0
        steps = 0
        max_test_steps = env_config["eval_env"]["max_episode_steps"]

        for step in range(max_test_steps):
            obs = test_env.get_observation()
            action, _ = model.predict(obs)
            test_env.step(action)

            reward = test_env.get_reward()
            total_reward += reward.mean().item()
            steps += 1

            # Check if done
            if hasattr(test_env, "get_success"):
                success = test_env.get_success()
                if success.any():
                    break

        avg_reward = total_reward / steps if steps > 0 else 0.0
        logger.info(
            "Test Results: %d steps, avg reward: %.4f, total: %.4f",
            steps,
            avg_reward,
            total_reward,
        )

        # Save test results
        test_results = {
            "environment": args.env,
            "algorithm": args.algorithm,
            "weight_file": args.weight,
            "steps": steps,
            "average_reward": avg_reward,
            "total_reward": total_reward,
        }

        with open(
            test_folder / f"test_results_{args.comment or 'default'}.json", "w"
        ) as f:
            json.dump(test_results, f, indent=2)

        logger.info("Testing completed and results saved.")
        return True

    except Exception as e:
        logger.exception("Testing failed: %s", e)
        return False


def main():
    """Main entry point"""
    args = parse_args()

    # Set random seed
    th.manual_seed(args.seed)

    try:
        # Configure logging once at entry if not already configured
        if not logging.getLogger().hasHandlers():
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
            )
        if args.train:
            success = run_training(args)
        else:
            success = run_testing(args)

        return 0 if success else 1

    except Exception as e:
        logger.exception("System failed: %s", e)
        return 1


if __name__ == "__main__":
    exit(main())

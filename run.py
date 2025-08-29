#!/usr/bin/env python3
"""
Unified Run System for VisFly-Eureka

This is the main entry point for training and testing environments with
algorithm configs, inspired by obj_track's run.py but integrated with
the Eureka pipeline.
"""

import sys
import os
import torch as th
import traceback
import argparse
from pathlib import Path

# Set matplotlib backend for headless servers
import matplotlib

matplotlib.use("Agg")

# Add paths (project root first, then VisFly)
sys.path.insert(0, str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "VisFly"))

# Import environments (use absolute paths to avoid VisFly conflicts)
import importlib.util
from pathlib import Path


def load_env_class(env_name, class_name):
    """Dynamically load environment class from specific file"""
    env_file = Path(__file__).parent / "envs" / f"{env_name}.py"
    spec = importlib.util.spec_from_file_location(class_name, env_file)
    module = importlib.util.module_from_spec(spec)
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
                print(
                    f"✅ Successfully injected reward function from {reward_function_path}"
                )
                return True
            else:
                print(f"❌ No get_reward function found in {reward_function_path}")
                return False

        except Exception as e:
            print(f"❌ Failed to inject reward function: {e}")
            traceback.print_exc()
            return False

    return False


def create_environment(env_name: str, env_config: dict, is_training: bool = True):
    """Create environment instance"""

    # Environment mapping
    env_classes = {
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


def run_training(args):
    """Execute training workflow"""

    try:
        # Load configurations
        env_config, alg_config = load_configs(args.env, args.algorithm)
        env_config, alg_config = apply_config_overrides(env_config, alg_config, args)

        # Setup save directory
        save_folder = Path(__file__).parent / "results" / "training" / args.env
        save_folder.mkdir(parents=True, exist_ok=True)
        save_folder = str(save_folder) + "/"

        # Create training environment
        train_env = create_environment(args.env, env_config, is_training=True)

        # Inject custom reward function if provided (Eureka integration)
        env_classes = {
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

        if args.reward_function_path:
            inject_reward_function(args.reward_function_path, env_classes[args.env])
            # Reset environment to apply new reward function
            train_env.reset()

        # Create algorithm
        model = create_algorithm(args.algorithm, alg_config, train_env, save_folder)

        # Load existing weights if specified
        if args.weight is not None:
            weight_path = save_folder + args.weight
            model.load(path=weight_path, env=train_env)

        # Training
        learning_params = alg_config["learn"]

        # Call learn method based on algorithm
        if args.algorithm in ["bptt", "shac"]:
            model.learn(**learning_params)
        elif args.algorithm == "ppo":
            model.learn(total_timesteps=int(learning_params["total_timesteps"]))

        model.save()

        return True

    except Exception as e:
        print(f"Training failed: {e}")
        traceback.print_exc()
        return False


def run_testing(args):
    """Execute testing workflow"""

    try:
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

        avg_reward = total_reward / steps
        print(
            f"Test Results: {steps} steps, avg reward: {avg_reward:.4f}, total: {total_reward:.4f}"
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

        return True

    except Exception as e:
        print(f"Testing failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Main entry point"""
    args = parse_args()

    # Set random seed
    th.manual_seed(args.seed)

    try:
        if args.train:
            success = run_training(args)
        else:
            success = run_testing(args)

        return 0 if success else 1

    except Exception as e:
        print(f"System failed: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())

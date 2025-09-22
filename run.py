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
from typing import Any, cast, List, Dict, Optional
import numpy as np
import time

# Set matplotlib backend for headless servers
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

try:
    import cv2
except ImportError:
    cv2 = None
    print("Warning: OpenCV not installed. Video recording disabled.")

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
CirclingEnv = load_env_class("CirclingEnv", "CirclingEnv")

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
    parser.set_defaults(no_video=False, no_plots=False)
    parser.add_argument(
        "--env",
        "-e",
        type=str,
        default="hover",
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
            "circling",
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

    # Video and visualization options (enabled by default in test mode)
    parser.add_argument(
        "--record_video",
        action="store_true",
        default=None,  # Will be set based on train mode
        help="Record videos during evaluation (default: True in test mode)",
    )
    parser.add_argument(
        "--plot_trajectories",
        action="store_true",
        default=None,  # Will be set based on train mode
        help="Generate trajectory plots during evaluation (default: True in test mode)",
    )
    parser.add_argument(
        "--no_video",
        action="store_true",
        help="Disable video recording in test mode",
    )
    parser.add_argument(
        "--no_plots",
        action="store_true",
        help="Disable trajectory plotting in test mode",
    )
    parser.add_argument(
        "--num_eval_episodes",
        type=int,
        default=None,
        help="Number of episodes to evaluate in test mode (default: from config or 10)",
    )
    parser.add_argument(
        "--video_fps",
        type=int,
        default=30,
        help="FPS for video recording",
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

    # Remove environment-specific parameters that aren't constructor args
    # These will be set as attributes after construction if needed
    extra_params = ["landing_target", "landing_tolerance", "wind_disturbance"]
    saved_params = {}
    for param in extra_params:
        if param in env_kwargs:
            saved_params[param] = env_kwargs.pop(param)

    # Convert target to tensor if it's a list
    if "target" in env_kwargs and isinstance(env_kwargs["target"], list):
        env_kwargs["target"] = th.tensor([env_kwargs["target"]])

    env = env_class(**env_kwargs)

    # Set extra parameters as attributes if they were present
    for param, value in saved_params.items():
        if param == "target" or param == "landing_target":
            # Convert to tensor if needed
            if isinstance(value, list):
                value = th.tensor([value] if not isinstance(value[0], list) else value)
        setattr(env, param, value)

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
        "circling": CirclingEnv,
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


def collect_trajectory_data(env, obs):
    """Collect trajectory data from environment state"""
    data = {}

    # Get position, velocity, orientation
    if hasattr(env, 'position'):
        data['position'] = env.position.detach().cpu().numpy() if hasattr(env.position, 'detach') else np.array(env.position)
    if hasattr(env, 'velocity'):
        data['velocity'] = env.velocity.detach().cpu().numpy() if hasattr(env.velocity, 'detach') else np.array(env.velocity)
    if hasattr(env, 'orientation'):
        data['orientation'] = env.orientation.detach().cpu().numpy() if hasattr(env.orientation, 'detach') else np.array(env.orientation)
    if hasattr(env, 'angular_velocity'):
        data['angular_velocity'] = env.angular_velocity.detach().cpu().numpy() if hasattr(env.angular_velocity, 'detach') else np.array(env.angular_velocity)
    if hasattr(env, 'target'):
        data['target'] = env.target.detach().cpu().numpy() if hasattr(env.target, 'detach') else np.array(env.target)

    return data


def create_trajectory_plot(trajectory_data: List[Dict], save_path: Path, episode_idx: int):
    """Create trajectory visualization plots"""
    if not trajectory_data:
        return

    # Extract time series data
    positions = np.array([d['position'] for d in trajectory_data if 'position' in d])
    velocities = np.array([d['velocity'] for d in trajectory_data if 'velocity' in d])

    if len(positions) == 0:
        return

    # Handle different position shapes: (T, 3) or (T, N, 3)
    if len(positions.shape) == 2:  # (T, 3) - single agent
        positions = positions[:, np.newaxis, :]  # Convert to (T, 1, 3)

    num_agents = positions.shape[1]
    timesteps = np.arange(len(positions))

    fig = plt.figure(figsize=(16, 12))

    # 3D Trajectory Plot
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    for agent_idx in range(min(4, num_agents)):  # Plot up to 4 agents
        x, y, z = positions[:, agent_idx, 0], positions[:, agent_idx, 1], positions[:, agent_idx, 2]

        ax1.plot(x, y, z, alpha=0.7, linewidth=1.5)
        ax1.scatter(x[0], y[0], z[0], color='green', s=50, marker='o', label=f'Start {agent_idx}')
        ax1.scatter(x[-1], y[-1], z[-1], color='red', s=50, marker='X', label=f'End {agent_idx}')

    # Plot target if available
    if trajectory_data[0].get('target') is not None:
        target = trajectory_data[0]['target']
        if len(target.shape) == 2:
            ax1.scatter(target[:, 0], target[:, 1], target[:, 2], c='blue', s=100, marker='*', label='Target')
        else:
            ax1.scatter(target[0], target[1], target[2], c='blue', s=100, marker='*', label='Target')

    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D Trajectory')
    ax1.legend()

    # Position over time
    ax2 = fig.add_subplot(2, 2, 2)
    mean_pos = positions.mean(axis=1)  # Average over agents
    std_pos = positions.std(axis=1) if num_agents > 1 else np.zeros_like(mean_pos)

    for i, label in enumerate(['X', 'Y', 'Z']):
        ax2.plot(timesteps, mean_pos[:, i], label=label, linewidth=2)
        if num_agents > 1:
            ax2.fill_between(timesteps, mean_pos[:, i] - std_pos[:, i],
                            mean_pos[:, i] + std_pos[:, i], alpha=0.2)
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Position (m)')
    ax2.set_title('Position vs Time')
    ax2.legend()
    ax2.grid(True)

    # Velocity over time
    if len(velocities) > 0:
        ax3 = fig.add_subplot(2, 2, 3)
        # Handle velocity shape
        if len(velocities.shape) == 2:  # (T, 3)
            velocities = velocities[:, np.newaxis, :]  # Convert to (T, 1, 3)

        mean_vel = velocities.mean(axis=1)  # Average over agents
        speed = np.linalg.norm(mean_vel, axis=1)

        ax3.plot(timesteps[:len(speed)], speed, linewidth=2, color='purple')
        ax3.set_xlabel('Time Step')
        ax3.set_ylabel('Speed (m/s)')
        ax3.set_title('Speed vs Time')
        ax3.grid(True)

    # Distance to target over time
    if trajectory_data[0].get('target') is not None:
        ax4 = fig.add_subplot(2, 2, 4)
        target = trajectory_data[0]['target']

        # Ensure target has correct shape
        if len(target.shape) == 1:
            target_pos = target[:3]
        elif len(target.shape) == 2:
            target_pos = target[0, :3] if target.shape[0] > 0 else target[:3]
        else:
            target_pos = target[:3]

        distances = []
        for pos_t in positions:  # pos_t shape: (N, 3)
            dist = np.linalg.norm(pos_t - target_pos[None, :], axis=1)  # Broadcast target
            distances.append(dist.mean())  # Average distance across agents

        ax4.plot(timesteps, distances, linewidth=2, color='orange')
        ax4.set_xlabel('Time Step')
        ax4.set_ylabel('Distance (m)')
        ax4.set_title('Distance to Target')
        ax4.grid(True)
        ax4.axhline(y=0.5, color='green', linestyle='--', alpha=0.5, label='Success Threshold')
        ax4.legend()

    plt.tight_layout()
    plot_path = save_path / f"trajectory_episode_{episode_idx:03d}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved trajectory plot: {plot_path}")


def record_video_frame(env, video_writer=None, video_params=None):
    """Record a video frame from environment render"""
    if cv2 is None or video_writer is None:
        return video_writer

    try:
        # Try to render the environment
        frame = env.render(mode='rgb_array') if hasattr(env, 'render') else None

        if frame is None:
            return video_writer

        # Handle different frame formats
        if isinstance(frame, th.Tensor):
            frame = frame.detach().cpu().numpy()

        # Ensure proper shape (H, W, 3)
        if len(frame.shape) == 4:  # (N, C, H, W) or (N, H, W, C)
            frame = frame[0]  # Take first agent

        if len(frame.shape) == 3:
            if frame.shape[0] in [1, 3, 4]:  # Channel first
                frame = np.transpose(frame, (1, 2, 0))
            if frame.shape[-1] == 1:  # Grayscale
                frame = np.repeat(frame, 3, axis=-1)
            elif frame.shape[-1] == 4:  # RGBA
                frame = frame[:, :, :3]

        # Normalize to uint8
        if frame.dtype != np.uint8:
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            else:
                frame = np.clip(frame, 0, 255).astype(np.uint8)

        # Convert RGB to BGR for OpenCV
        if len(frame.shape) == 3 and frame.shape[-1] == 3:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            frame_bgr = frame

        # Initialize video writer if needed
        if video_writer == 'init':
            height, width = frame_bgr.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_path = video_params['path']
            fps = video_params['fps']
            video_writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
            if not video_writer.isOpened():
                logger.warning(f"Failed to open video writer for {video_path}")
                return None

        # Write frame
        if video_writer is not None and video_writer.isOpened():
            video_writer.write(frame_bgr)

        return video_writer

    except Exception as e:
        logger.debug(f"Frame recording failed: {e}")
        return video_writer


def run_testing(args):
    """Execute enhanced testing workflow with video and visualization"""

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

        # Create subdirectories for videos and plots if needed
        if args.record_video:
            video_folder = test_folder / "videos"
            video_folder.mkdir(exist_ok=True)

        if args.plot_trajectories:
            plot_folder = test_folder / "plots"
            plot_folder.mkdir(exist_ok=True)

        # Create evaluation environment
        test_env = create_environment(args.env, env_config, is_training=False)

        # Load trained model
        if args.weight is None:
            raise ValueError("Weight file required for testing mode")

        weight_path = save_folder / args.weight

        # Create algorithm and load weights
        algorithm_classes = {"bptt": BPTT, "ppo": PPO, "shac": SHAC}
        model = algorithm_classes[args.algorithm].load(str(weight_path), env=test_env)

        # Get number of evaluation episodes from config or args
        # Priority: command-line arg > algorithm config > env config > default (10)
        num_episodes = args.num_eval_episodes
        if num_episodes is None:
            # Try different config locations
            # 1. Algorithm config (test section)
            if "test" in alg_config and "episodes" in alg_config["test"]:
                num_episodes = alg_config["test"]["episodes"]
            # 2. Algorithm config (learn section)
            elif "evaluation_episodes" in alg_config.get("learn", {}):
                num_episodes = alg_config["learn"]["evaluation_episodes"]
            # 3. Default to 10 as in main.py's config
            else:
                num_episodes = 10

        max_test_steps = env_config["eval_env"]["max_episode_steps"]

        all_results = []
        success_count = 0

        logger.info(f"Running {num_episodes} evaluation episodes...")

        for episode_idx in range(num_episodes):
            logger.info(f"Episode {episode_idx + 1}/{num_episodes}")

            # Reset environment
            obs = test_env.reset()

            # Episode tracking
            episode_reward = 0
            episode_steps = 0
            trajectory_data = []
            video_writer = None

            # Initialize video recording if enabled
            if args.record_video and cv2 is not None:
                video_path = video_folder / f"episode_{episode_idx:03d}.mp4"
                video_params = {'path': video_path, 'fps': args.video_fps}
                video_writer = 'init'  # Will be initialized on first frame

            # Episode loop
            done = False
            for step in range(max_test_steps):
                # Get observation and predict action
                if hasattr(test_env, 'get_observation'):
                    obs = test_env.get_observation()

                if hasattr(model, 'predict'):
                    action = model.predict(obs, deterministic=True)
                    if isinstance(action, tuple):
                        action = action[0]
                else:
                    action = test_env.action_space.sample()

                # Step environment
                if hasattr(test_env, 'step'):
                    test_env.step(action)
                else:
                    obs, reward, done, info = test_env.step(action)

                # Get reward
                if hasattr(test_env, 'get_reward'):
                    reward = test_env.get_reward()
                    if isinstance(reward, dict):
                        # Handle dict rewards (e.g., from environments with individual reward components)
                        if "reward" in reward:
                            reward_value = float(reward["reward"].mean().item()) if isinstance(reward["reward"], th.Tensor) else float(reward["reward"])
                        else:
                            # Sum all reward components
                            reward_value = sum(float(v.mean().item()) if isinstance(v, th.Tensor) else float(v) for v in reward.values())
                    elif isinstance(reward, th.Tensor):
                        reward_value = reward.mean().item()
                    elif isinstance(reward, (list, np.ndarray)):
                        reward_value = float(np.mean(reward))
                    else:
                        reward_value = float(reward)
                else:
                    reward_value = float(reward) if not isinstance(reward, (list, np.ndarray)) else float(np.mean(reward))

                episode_reward += reward_value
                episode_steps += 1

                # Collect trajectory data if enabled
                if args.plot_trajectories:
                    traj_data = collect_trajectory_data(test_env, obs)
                    if traj_data:
                        trajectory_data.append(traj_data)

                # Record video frame if enabled
                if args.record_video:
                    video_writer = record_video_frame(test_env, video_writer, video_params)

                # Check termination
                if hasattr(test_env, 'get_success'):
                    success = test_env.get_success()
                    if isinstance(success, th.Tensor):
                        done = success.any().item()
                    else:
                        done = bool(np.any(success))

                if done:
                    break

            # Check final success
            episode_success = False
            if hasattr(test_env, 'get_success'):
                success = test_env.get_success()
                if isinstance(success, th.Tensor):
                    episode_success = success.any().item()
                else:
                    episode_success = bool(np.any(success))

            if episode_success:
                success_count += 1

            # Save video
            if video_writer is not None and hasattr(video_writer, 'release'):
                video_writer.release()
                logger.info(f"Saved video: {video_path}")

            # Create trajectory plot
            if args.plot_trajectories and trajectory_data:
                create_trajectory_plot(trajectory_data, plot_folder, episode_idx)

            # Store episode results
            episode_result = {
                "episode": episode_idx,
                "steps": episode_steps,
                "total_reward": episode_reward,
                "average_reward": episode_reward / episode_steps if episode_steps > 0 else 0,
                "success": episode_success
            }
            all_results.append(episode_result)

            logger.info(
                f"Episode {episode_idx + 1}: steps={episode_steps}, "
                f"reward={episode_reward:.4f}, success={episode_success}"
            )

        # Compute aggregate statistics
        total_steps = sum(r["steps"] for r in all_results)
        total_reward = sum(r["total_reward"] for r in all_results)
        success_rate = success_count / num_episodes
        avg_episode_reward = total_reward / num_episodes
        avg_episode_steps = total_steps / num_episodes

        logger.info("\n" + "="*50)
        logger.info("EVALUATION SUMMARY")
        logger.info(f"Episodes: {num_episodes}")
        logger.info(f"Success Rate: {success_rate:.2%} ({success_count}/{num_episodes})")
        logger.info(f"Average Episode Reward: {avg_episode_reward:.4f}")
        logger.info(f"Average Episode Steps: {avg_episode_steps:.1f}")
        logger.info("="*50)

        # Save detailed test results
        test_results = {
            "environment": args.env,
            "algorithm": args.algorithm,
            "weight_file": args.weight,
            "num_episodes": num_episodes,
            "success_rate": success_rate,
            "success_count": success_count,
            "average_episode_reward": avg_episode_reward,
            "average_episode_steps": avg_episode_steps,
            "total_reward": total_reward,
            "total_steps": total_steps,
            "video_recorded": args.record_video,
            "trajectory_plotted": args.plot_trajectories,
            "episodes": all_results
        }

        results_file = test_folder / f"test_results_{args.comment or 'default'}_{time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, "w") as f:
            json.dump(test_results, f, indent=2)

        logger.info(f"\nResults saved to: {results_file}")

        if args.record_video and cv2 is not None:
            logger.info(f"Videos saved to: {video_folder}")

        if args.plot_trajectories:
            logger.info(f"Trajectory plots saved to: {plot_folder}")

        return True

    except Exception as e:
        logger.exception("Testing failed: %s", e)
        return False


def main():
    """Main entry point"""
    args = parse_args()

    # Set defaults for visualization options based on mode
    if not args.train:  # Test mode
        # Enable video and plots by default in test mode unless explicitly disabled
        if args.record_video is None:
            args.record_video = not args.no_video
        if args.plot_trajectories is None:
            args.plot_trajectories = not args.no_plots
    else:  # Training mode
        # Disable by default in training mode
        if args.record_video is None:
            args.record_video = False
        if args.plot_trajectories is None:
            args.plot_trajectories = False

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

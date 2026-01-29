"""
Standalone evaluation worker invoked by SubprocessRewardEvaluator.

This script runs in a separate Python process to train and evaluate a model
with an injected reward function, returning JSON results via an output file.

Key behaviors:
- Adds project_root and project_root/VisFly to sys.path (from config)
- Dynamically imports the specified environment class
- Injects the provided reward function (expects a get_reward(self) function)
- Supports multiple training algorithms: BPTT, PPO, SHAC
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
import math
import inspect
import types
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
import torch
import torch as th
import numpy as np

import cv2  # type: ignore

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Import FigFashion from VisFly for 3x3 plots
visfly_path = Path(__file__).resolve().parents[2] / "VisFly"
if str(visfly_path) not in sys.path:
    sys.path.insert(0, str(visfly_path))
from VisFly.utils.FigFashion.FigFashion import FigFon
from VisFly.utils.maths import Quaternion

# Import algorithms.BPTT_series.BPTT at top level
from algorithms.BPTT_series.BPTT import BPTT


def ensure_video_writer(writer: Optional[Any], frame: np.ndarray, path: Path, fps: int) -> Any:
    """Ensure video writer is initialized using run.py approach"""
    if writer is not None:
        return writer
    height, width = frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer at {path}")
    return writer


def record_frame(env: Any, writer: Optional[Any], path: Path, fps: int) -> Optional[Any]:
    """Record frame using run.py approach"""
    try:
        frame = env.render(mode="rgb_array")
    except TypeError:
        frame = env.render()

    if isinstance(frame, dict):
        frame = next(iter(frame.values()))
    frame = np.asarray(frame)

    if frame.ndim == 4:
        frame = frame[0]
    if frame.shape[0] in (1, 3, 4) and frame.shape[-1] not in (3, 4):
        frame = np.moveaxis(frame, 0, -1)
    if frame.shape[-1] == 4:
        frame = frame[..., :3]
    if frame.dtype != np.uint8:
        frame = np.clip(frame * 255 if frame.max() <= 1.0 else frame, 0, 255).astype(np.uint8)

    writer = ensure_video_writer(writer, frame, path, fps)
    writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    return writer


def snapshot_env(env: Any) -> Dict[str, np.ndarray]:
    """Capture environment state for trajectory plotting"""
    def to_numpy(value: Any) -> np.ndarray:
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy().copy()
        return np.array(value, copy=True)

    if hasattr(env, "target"):
        target_attr = env.target
    elif hasattr(env, "target_position"):
        target_attr = env.target_position
    else:
        target_attr = None

    return {
        "position": to_numpy(env.position),
        "velocity": to_numpy(env.velocity),
        "orientation": to_numpy(env.orientation),
        "angular_velocity": to_numpy(env.angular_velocity),
        "target": to_numpy(target_attr),
    }


def _eval_artifact_basename(num_runs: int, episode_idx: int) -> str:
    """Basename for eval artifacts: 'eval' when a single run, else 'episode_XXX'."""
    return "eval" if num_runs == 1 else f"episode_{episode_idx:03d}"


def plot_trajectory(
    trajectory: List[Dict[str, np.ndarray]],
    output: Path,
    episode_idx: int,
    logger: logging.Logger,
    plot_basename: Optional[str] = None,
) -> None:
    """Plot trajectory using 3x3 subplot layout with VisFly FigFashion helper."""
    positions = np.stack([step["position"] for step in trajectory])
    velocities = np.stack([step["velocity"] for step in trajectory])
    orientations = np.stack([step.get("orientation", np.zeros((1, 4))) for step in trajectory])
    angular_velocities = np.stack([step.get("angular_velocity", np.zeros((1, 3))) for step in trajectory])

    if positions.ndim == 2:
        positions = positions[:, None, :]
    if velocities.ndim == 2:
        velocities = velocities[:, None, :]
    if orientations.ndim == 2:
        orientations = orientations[:, None, :]
    if angular_velocities.ndim == 2:
        angular_velocities = angular_velocities[:, None, :]

    timesteps = np.arange(len(positions))
    num_agents = positions.shape[1]

    target = trajectory[0].get("target")
    if target is not None:
        target = np.asarray(target)
        if target.ndim == 1:
            target = target.reshape(1, -1)

    # Use FigFashion for 3x3 layout
    fig, axes = FigFon.get_figure_axes(SubFigSize=(3, 3), Column=2, HeightScale=1.2)
    axes_flat = axes.flatten()
    
    # Replace first axes with 3D projection for trajectory plot
    axes_flat[0].remove()
    axes_flat[0] = fig.add_subplot(3, 3, 1, projection="3d")

    # Compute mean values across agents
    mean_pos = positions.mean(axis=1)
    mean_vel = velocities.mean(axis=1)
    mean_angvel = angular_velocities.mean(axis=1)
    mean_ori = orientations.mean(axis=1)

    # Convert quaternion orientations to Euler angles (roll, pitch, yaw)
    # Orientation format: [w, x, y, z] from dynamics.orientation (via toTensor())
    euler_angles = np.zeros((len(timesteps), 3))  # [roll, pitch, yaw]
    for i in range(len(timesteps)):
        q_arr = mean_ori[i]  # Shape: (4,) numpy array [w, x, y, z]
        if q_arr.shape[0] == 4:
            # Convert to torch tensor and create Quaternion object
            # Quaternion expects [w, x, y, z] format
            q = Quaternion(
                w=th.tensor(q_arr[0], dtype=th.float32),
                x=th.tensor(q_arr[1], dtype=th.float32),
                y=th.tensor(q_arr[2], dtype=th.float32),
                z=th.tensor(q_arr[3], dtype=th.float32)
            )
            euler = q.toEuler(order="zyx")  # Returns [roll, pitch, yaw] tensor
            euler_angles[i] = euler.detach().cpu().numpy()
        else:
            # Fallback: skip if invalid shape
            logger.warning(f"Invalid orientation shape at timestep {i}: {q_arr.shape}")

    # Plot 1: 3D Trajectory
    ax_3d = axes_flat[0]
    for agent_idx in range(min(num_agents, 4)):
        x, y, z = positions[:, agent_idx, :].T
        ax_3d.plot(x, y, z, linewidth=1.5)
        ax_3d.scatter(x[0], y[0], z[0], color="green", marker="o", s=50, label="Start" if agent_idx == 0 else "")
        ax_3d.scatter(x[-1], y[-1], z[-1], color="red", marker="x", s=60, label="End" if agent_idx == 0 else "")
    if target is not None:
        ax_3d.scatter(target[:, 0], target[:, 1], target[:, 2], color="blue", marker="*", s=120, label="Target")
    ax_3d.set_xlabel("X (m)")
    ax_3d.set_ylabel("Y (m)")
    ax_3d.set_zlabel("Z (m)")
    ax_3d.set_title("3D Trajectory")
    ax_3d.legend()

    # Plot 2: Position (X, Y, Z)
    axes_flat[1].plot(timesteps, mean_pos[:, 0], label="X", color="blue", linewidth=1.5)
    axes_flat[1].plot(timesteps, mean_pos[:, 1], label="Y", color="green", linewidth=1.5)
    axes_flat[1].plot(timesteps, mean_pos[:, 2], label="Z", color="red", linewidth=1.5)
    axes_flat[1].set_xlabel("Time Step")
    axes_flat[1].set_ylabel("Position (m)")
    axes_flat[1].legend()
    axes_flat[1].grid(True)
    axes_flat[1].set_title("Position")

    # Plot 3: Velocity (Vx, Vy, Vz)
    axes_flat[2].plot(timesteps, mean_vel[:, 0], label="Vx", color="blue", linewidth=1.5)
    axes_flat[2].plot(timesteps, mean_vel[:, 1], label="Vy", color="green", linewidth=1.5)
    axes_flat[2].plot(timesteps, mean_vel[:, 2], label="Vz", color="red", linewidth=1.5)
    axes_flat[2].set_xlabel("Time Step")
    axes_flat[2].set_ylabel("Velocity (m/s)")
    axes_flat[2].legend()
    axes_flat[2].grid(True)
    axes_flat[2].set_title("Velocity")

    # Plot 4: Angular Velocity (ωx, ωy, ωz)
    axes_flat[3].plot(timesteps, mean_angvel[:, 0], label="ωx", color="blue", linewidth=1.5)
    axes_flat[3].plot(timesteps, mean_angvel[:, 1], label="ωy", color="green", linewidth=1.5)
    axes_flat[3].plot(timesteps, mean_angvel[:, 2], label="ωz", color="red", linewidth=1.5)
    axes_flat[3].set_xlabel("Time Step")
    axes_flat[3].set_ylabel("Angular Velocity (rad/s)")
    axes_flat[3].legend()
    axes_flat[3].grid(True)
    axes_flat[3].set_title("Angular Velocity")

    # Plot 5: Euler Angles (Roll, Pitch, Yaw)
    axes_flat[4].plot(timesteps, np.degrees(euler_angles[:, 0]), label="Roll", color="blue", linewidth=1.5)
    axes_flat[4].plot(timesteps, np.degrees(euler_angles[:, 1]), label="Pitch", color="green", linewidth=1.5)
    axes_flat[4].plot(timesteps, np.degrees(euler_angles[:, 2]), label="Yaw", color="red", linewidth=1.5)
    axes_flat[4].set_xlabel("Time Step")
    axes_flat[4].set_ylabel("Euler Angles (deg)")
    axes_flat[4].legend()
    axes_flat[4].grid(True)
    axes_flat[4].set_title("Orientation (Euler)")

    # Plot 6: Distance to Target
    if target is not None:
        target_pos = target[0] if target.ndim == 2 else target
        distances = np.linalg.norm(positions - target_pos, axis=2).mean(axis=1)
        axes_flat[5].plot(timesteps, distances, color="orange", linewidth=1.5, label="Distance")
        axes_flat[5].set_xlabel("Time Step")
        axes_flat[5].set_ylabel("Distance (m)")
        axes_flat[5].legend()
        axes_flat[5].grid(True)
        axes_flat[5].set_title("Distance to Target")
    else:
        distances = np.linalg.norm(positions, axis=2).mean(axis=1)
        axes_flat[5].plot(timesteps, distances, color="orange", linewidth=1.5, label="Distance from Origin")
        axes_flat[5].set_xlabel("Time Step")
        axes_flat[5].set_ylabel("Distance (m)")
        axes_flat[5].legend()
        axes_flat[5].grid(True)
        axes_flat[5].set_title("Distance from Origin")

    # Plot 7: Speed Magnitude
    speed_magnitude = np.linalg.norm(mean_vel, axis=1)
    axes_flat[6].plot(timesteps, speed_magnitude, color="purple", linewidth=1.5, label="Speed")
    axes_flat[6].set_xlabel("Time Step")
    axes_flat[6].set_ylabel("Speed (m/s)")
    axes_flat[6].legend()
    axes_flat[6].grid(True)
    axes_flat[6].set_title("Speed Magnitude")

    # Plot 8: Angular Velocity Magnitude
    angvel_magnitude = np.linalg.norm(mean_angvel, axis=1)
    axes_flat[7].plot(timesteps, angvel_magnitude, color="purple", linewidth=1.5, label="|ω|")
    axes_flat[7].set_xlabel("Time Step")
    axes_flat[7].set_ylabel("Angular Velocity (rad/s)")
    axes_flat[7].legend()
    axes_flat[7].grid(True)
    axes_flat[7].set_title("Angular Velocity Magnitude")

    # Plot 9: Position Error (if target exists) or Position Magnitude
    if target is not None:
        target_pos = target[0] if target.ndim == 2 else target
        pos_error = positions - target_pos
        mean_error = pos_error.mean(axis=1)
        axes_flat[8].plot(timesteps, mean_error[:, 0], label="Ex", color="blue", linewidth=1.5)
        axes_flat[8].plot(timesteps, mean_error[:, 1], label="Ey", color="green", linewidth=1.5)
        axes_flat[8].plot(timesteps, mean_error[:, 2], label="Ez", color="red", linewidth=1.5)
        axes_flat[8].set_xlabel("Time Step")
        axes_flat[8].set_ylabel("Position Error (m)")
        axes_flat[8].legend()
        axes_flat[8].grid(True)
        axes_flat[8].set_title("Position Error")
    else:
        pos_magnitude = np.linalg.norm(mean_pos, axis=1)
        axes_flat[8].plot(timesteps, pos_magnitude, color="purple", linewidth=1.5, label="Position Magnitude")
        axes_flat[8].set_xlabel("Time Step")
        axes_flat[8].set_ylabel("Position (m)")
        axes_flat[8].legend()
        axes_flat[8].grid(True)
        axes_flat[8].set_title("Position Magnitude")

    plt.tight_layout()
    name = plot_basename if plot_basename is not None else f"episode_{episode_idx:03d}"
    fig_path = output / f"trajectory_{name}.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved trajectory plot: {fig_path}")


def _mean(values: List[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _std(values: List[float]) -> float:
    if len(values) <= 1:
        return 0.0
    avg = _mean(values)
    variance = sum((v - avg) ** 2 for v in values) / len(values)
    return float(variance ** 0.5)


def _extract_rgb_frame(frame: Any) -> Optional["np.ndarray"]:
    """Normalize different frame layouts to HxWx3 RGB uint8 arrays."""
    if frame is None:
        return None

    # Handle lists/tuples coming from vectorized envs
    if isinstance(frame, (list, tuple)):
        if len(frame) == 0:
            return None
        frame = frame[0]

    if isinstance(frame, dict):
        # Prefer the first ndarray entry
        for value in frame.values():
            candidate = _extract_rgb_frame(value)
            if candidate is not None:
                return candidate
        return None

    if not isinstance(frame, np.ndarray):
        return None

    if frame.ndim == 4:
        frame = frame[0]

    if frame.ndim == 3 and frame.shape[0] in (1, 3, 4) and frame.shape[0] != frame.shape[-1]:
        # Channel-first -> channel-last
        frame = np.transpose(frame, (1, 2, 0))

    if frame.ndim == 2:
        frame = np.stack([frame] * 3, axis=-1)

    # Convert grayscale or RGBA to RGB
    if frame.shape[-1] == 1:
        frame = np.repeat(frame, 3, axis=-1)
    elif frame.shape[-1] == 4:
        # Drop alpha channel
        frame = frame[..., :3]

    frame = np.clip(frame, 0, 255)
    if frame.dtype != np.uint8:
        frame = frame.astype(np.uint8)

    if frame.shape[-1] != 3:
        return None

    return frame


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
    # Use same execution context as run.py
    exec_globals: Dict[str, Any] = {
        "torch": th,
        "th": th,
        "np": np,
        "numpy": np,
        "math": math,
    }
    
    # Extract helper functions from environment module
    # These are functions defined at module level (not class methods, not imported)
    try:
        env_module = inspect.getmodule(env_class)
        if env_module is not None:
            for name in dir(env_module):
                # Skip private attributes and things already in exec_globals
                if name.startswith('_') or name in exec_globals:
                    continue
                
                try:
                    obj = getattr(env_module, name)
                    # Include only functions defined in this module (not imported)
                    if inspect.isfunction(obj) and inspect.getmodule(obj) == env_module:
                        exec_globals[name] = obj
                        logger.debug(f"Added helper function '{name}' from environment module to exec context")
                except (AttributeError, TypeError):
                    # Skip attributes that can't be inspected
                    continue
    except Exception as e:
        logger.warning(f"Failed to extract helper functions from environment module: {e}")
        # Continue anyway, as helper functions may not exist
    
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


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def evaluate_main(config_path: Path, output_path: Path) -> int:
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    logger = logging.getLogger(__name__)

    # Initialize variables early to avoid locals() checks
    cfg = None
    identifier = "unknown"
    env: Optional[Any] = None
    env_eval: Optional[Any] = None

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
        if eval_episodes < 1:
            raise ValueError(f"evaluation_episodes must be >= 1, got {eval_episodes}")

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
        log_interval = learn_params.get("log_interval", 20)  # Default to 20 if not specified
        
        logger.info(f"Training configuration: {train_steps} timesteps, log_interval={log_interval}")
        
        # Keep devices separate - env and algorithm use their configured devices.
        # When parent fallback to CPU (GPU full), effective_algorithm_device is set in config.
        env_device = env_config["device"]
        alg_device = optimization_config.get("effective_algorithm_device") or alg_params["device"]
        alg_params["device"] = alg_device
        logger.info(f"Environment device: {env_device}, Algorithm device: {alg_device}")

        # Configure requires_grad and tensor_output based on algorithm type
        # BPTT and SHAC require gradients and tensor output (droneGymEnv enforces requires_grad -> tensor_output)
        if algorithm in ("bptt", "shac"):
            env_config = {**env_config, "requires_grad": True, "tensor_output": True}
            logger.info(f"requires_grad=True, tensor_output=True for {algorithm.upper()}")
        elif algorithm == "ppo":
            env_config = {**env_config, "requires_grad": False}
            logger.info("requires_grad=False for PPO")
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

        # Create training environment (will inherit injected reward function)
        logger.info("[stage] constructing training env ...")
        env = EnvClass(**env_config)
        logger.info("[stage] training env constructed with injected reward")

        # Note: Do NOT call env.reset() here - let the algorithm handle it properly


        # Dynamically import and instantiate algorithm
        logger.info(f"[stage] instantiating {algorithm.upper()} algorithm ...")

        if algorithm == "bptt":
            from algorithms.BPTT_series.BPTT import BPTT
            alg_class = BPTT
        elif algorithm == "ppo":
            from VisFly.utils.algorithms.PPO import PPO
            alg_class = PPO
        elif algorithm == "shac":
            from algorithms.BPTT_series.SHAC import SHAC
            alg_class = SHAC
        else:
            # This should never happen due to earlier check
            raise ValueError(f"Unsupported algorithm: {algorithm}")

        alg_params["env"] = env
        alg_params.setdefault("policy", "SimplePolicy")

        # Add TensorBoard logging support
        tensorboard_dir = Path(output_dir) / "tensorboard"
        tensorboard_dir.mkdir(exist_ok=True)
        alg_params["tensorboard_log"] = str(tensorboard_dir)
        logger.info(f"TensorBoard logs will be saved to: {tensorboard_dir}")

        model = alg_class(**alg_params)
        logger.info(f"[stage] {algorithm.upper()} algorithm instantiated")

        # Train
        logger.info(f"[stage] starting training for {train_steps} steps ...")
        t0 = time.time()
        
        # Track GPU memory (only on CUDA)
        peak_memory_mb = 0
        device_str = str(alg_device).lower()
        use_cuda = torch.cuda.is_available() and "cuda" in device_str
        if use_cuda:
            gpu_id = int(device_str.split(":")[1]) if ":" in device_str else 0
            torch.cuda.reset_peak_memory_stats(gpu_id)

        # Prepare training arguments
        train_args = {"total_timesteps": int(train_steps)}
        if log_interval is not None:
            train_args["log_interval"] = log_interval

        # Use getattr to avoid static type check issues
        getattr(model, "learn")(**train_args)
        training_time = time.time() - t0

        # Get peak GPU memory usage
        if use_cuda:
            peak_memory_mb = torch.cuda.max_memory_allocated(gpu_id) // (1024**2)
            final_memory = torch.cuda.memory_allocated(gpu_id) // (1024**2)
            logger.info(f"Peak GPU memory: {peak_memory_mb}MB, Final: {final_memory}MB")
        else:
            logger.info(f"Training completed in {training_time:.2f}s (CPU mode, no GPU stats)")
        
        logger.info(f"Training completed in {training_time:.2f}s")
        
        # Save trained model
        model_save_path = Path(output_dir) / "trained_model.zip"
        model.save(str(model_save_path))
        logger.info(f"Saved trained model to: {model_save_path}")

        # Create evaluation env (requires_grad False)
        eval_env_config = {**eval_env_config, "requires_grad": False}
        logger.info("[stage] constructing eval env ...")
        env_eval = EnvClass(**eval_env_config)
        logger.info("[stage] eval env constructed (inherits injected reward from class)")

        # Run evaluation episodes with detailed statistics
        requested_eval_episodes = int(eval_episodes)
        eval_runs = max(1, requested_eval_episodes)  # Ensure at least 1 episode
        logger.info(
            f"[stage] evaluating trained policy over {eval_runs} episodes"
        )

        success_flags: List[bool] = []
        collision_flags: List[bool] = []
        episode_lengths: List[float] = []
        episode_rewards: List[float] = []
        final_distances: List[float] = []
        episode_details: List[Dict[str, Any]] = []
        saved_videos: List[str] = []

        max_steps = int(eval_env_config["max_episode_steps"])
        record_video = bool(optimization_config.get("record_video", False))
        video_enabled = record_video and bool(eval_env_config.get("visual", False))
        video_fps = float(eval_env_config.get("video_fps", 30.0))
        video_dir = Path(output_dir) / "videos"
        plots_dir = Path(output_dir) / "plots"

        # Verbose diagnostics for video settings
        logger.info(
            f"[video] requested={record_video}, eval_env.visual={bool(eval_env_config.get('visual', False))}, "
            f"enabled={video_enabled}, fps={video_fps}, dir={video_dir}"
        )
        if record_video and not bool(eval_env_config.get("visual", False)):
            logger.warning("[video] Disabled because eval_env.visual is False. Enable visual=true under eval_env.")
        if video_enabled:
            video_dir.mkdir(parents=True, exist_ok=True)

        # Create plots directory for trajectory plots
        plots_dir.mkdir(parents=True, exist_ok=True)

        model.policy.set_training_mode(False)

        for episode_idx in range(eval_runs):
            obs = env_eval.reset()
            done_flag = False
            step_count = 0
            episode_reward = 0.0
            last_info = None
            video_writer = None
            artifact_basename = _eval_artifact_basename(eval_runs, episode_idx)
            video_path = video_dir / f"{artifact_basename}.mp4" if video_enabled else None

            # Initialize trajectory tracking
            trajectory = [snapshot_env(env_eval)]

            while step_count < max_steps and not done_flag:
                with torch.no_grad():
                    if hasattr(model, "predict"):
                        action = model.predict(obs, deterministic=True)
                        if isinstance(action, tuple):
                            action = action[0]
                    else:
                        action = env_eval.action_space.sample()

                obs, reward, done_arr, info = env_eval.step(action, is_test=True)
                last_info = info

                if isinstance(reward, th.Tensor):
                    reward_value = float(reward.mean().item())
                elif isinstance(reward, np.ndarray):
                    reward_value = float(np.mean(reward))
                else:
                    reward_value = float(reward)
                episode_reward += reward_value

                # Continue episode until all agents are done (not just any agent)
                # This allows remaining agents to continue even if one agent fails
                if isinstance(done_arr, (np.ndarray, list)):
                    done_flag = bool(np.all(done_arr))
                elif isinstance(done_arr, th.Tensor):
                    done_flag = bool(done_arr.all().item())
                else:
                    done_flag = bool(done_arr)

                # Capture trajectory data after each step
                trajectory.append(snapshot_env(env_eval))

                # Record video frame using run.py approach
                if video_enabled:
                    video_writer = record_frame(env_eval, video_writer, video_path, 30)

                step_count += 1

            if video_writer is not None:
                video_writer.release()
                saved_videos.append(str(video_path))
                logger.info(f"[video] Saved {video_path}")

            # Create trajectory plot for this episode
            plot_trajectory(trajectory, plots_dir, episode_idx, logger, plot_basename=artifact_basename)

            # Save trajectory data as npz file for LLM analysis
            trajectory_dir = Path(output_dir) / "trajectories"
            trajectory_dir.mkdir(parents=True, exist_ok=True)
            trajectory_npz_path = trajectory_dir / f"{artifact_basename}.npz"
            
            # Convert trajectory list to numpy arrays
            trajectory_data = {
                "positions": np.stack([step["position"] for step in trajectory]),
                "velocities": np.stack([step["velocity"] for step in trajectory]),
                "orientations": np.stack([step.get("orientation", np.zeros((1, 4))) for step in trajectory]),
                "angular_velocities": np.stack([step.get("angular_velocity", np.zeros((1, 3))) for step in trajectory]),
            }
            if trajectory[0].get("target") is not None:
                trajectory_data["target"] = np.asarray(trajectory[0]["target"])
            
            np.savez_compressed(trajectory_npz_path, **trajectory_data)
            logger.info(f"Saved trajectory data to {trajectory_npz_path}")

            success_tensor = env_eval.get_success()
            if isinstance(success_tensor, th.Tensor):
                success_bool = bool(success_tensor.any().item())
            else:
                success_bool = bool(success_tensor)

            if not success_bool and last_info:
                success_bool = any(
                    info.get("is_success", False) for info in last_info if isinstance(info, dict)
                )

            success_flags.append(success_bool)

            collision_flag = bool(env_eval.is_collision.any().item())
            collision_flags.append(collision_flag)

            episode_lengths.append(float(step_count))
            episode_rewards.append(float(episode_reward))

            target_for_distance = getattr(env_eval, "target", None)
            if target_for_distance is None:
                target_for_distance = getattr(env_eval, "target_position", None)

            if target_for_distance is not None:
                if isinstance(target_for_distance, th.Tensor):
                    target_tensor = target_for_distance
                else:
                    target_tensor = th.tensor(target_for_distance, device=env_eval.position.device, dtype=env_eval.position.dtype)

                if target_tensor.ndim == 1:
                    target_tensor = target_tensor.unsqueeze(0)

                distance_tensor = (env_eval.position - target_tensor).norm(dim=1)
                final_distance = float(distance_tensor.mean().item())
                final_distances.append(final_distance)
            else:
                final_distance = float("nan")
                final_distances.append(final_distance)

            time_truncated = False
            if last_info:
                time_truncated = any(
                    info.get("TimeLimit.truncated", False)
                    for info in last_info
                    if isinstance(info, dict)
                )
            if not time_truncated and step_count >= max_steps:
                time_truncated = True

            termination_reason = (
                "success"
                if success_bool
                else ("timeout" if time_truncated else "failure")
            )

            episode_record = {
                "episode_index": episode_idx,
                "steps": float(step_count),
                "episode_reward": float(episode_reward),
                "success": success_bool,
                "collision": collision_flag,
                "termination_reason": termination_reason,
                "final_distance_to_target": final_distance,
                "video_path": str(video_path) if video_path else None,
                "trajectory_path": str(trajectory_npz_path),
            }

            episode_details.append(episode_record)

        success_count = sum(1 for flag in success_flags if flag)
        success_rate = success_count / max(1, len(success_flags))

        aggregate_statistics: Dict[str, Any] = {
            "requested_evaluation_episodes": requested_eval_episodes,
            "actual_evaluation_episodes": len(success_flags),
            "success_count": success_count,
            "success_rate": success_rate,
            "mean_episode_length": _mean(episode_lengths),
            "std_episode_length": _std(episode_lengths),
            "min_episode_length": min(episode_lengths) if episode_lengths else 0.0,
            "max_episode_length": max(episode_lengths) if episode_lengths else 0.0,
            "mean_episode_reward": _mean(episode_rewards),
            "std_episode_reward": _std(episode_rewards),
            "min_episode_reward": min(episode_rewards) if episode_rewards else 0.0,
            "max_episode_reward": max(episode_rewards) if episode_rewards else 0.0,
            "collision_count": sum(1 for flag in collision_flags if flag),
        }

        if final_distances:
            aggregate_statistics.update(
                {
                    "mean_final_distance": _mean(final_distances),
                    "std_final_distance": _std(final_distances),
                    "min_final_distance": min(final_distances),
                    "max_final_distance": max(final_distances),
                }
            )

        avg_len = aggregate_statistics["mean_episode_length"]
        avg_final_reward = aggregate_statistics["mean_episode_reward"]

        aggregate_statistics["videos_recorded"] = len(saved_videos)
        aggregate_statistics["success_metric"] = (
            "env.get_success" if hasattr(env_eval, "get_success") else "episode_reward>0"
        )

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
            "peak_memory_mb": peak_memory_mb,
            "model_path": str(model_save_path),
            "evaluation_runs": len(success_flags),
            "aggregate_statistics": aggregate_statistics,
            "episode_statistics": episode_details,
            "video_paths": saved_videos,
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
                        "identifier": identifier,
                    },
                    f,
                    indent=2,
                )
        except Exception:
            # If even writing fails, log to stderr
            logging.error(tb)
        return 0
    finally:
        if env_eval is not None:
            try:
                env_eval.close()
            except Exception:
                logger.debug("Failed to close evaluation environment cleanly")
        if env is not None:
            try:
                env.close()
            except Exception:
                logger.debug("Failed to close training environment cleanly")


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

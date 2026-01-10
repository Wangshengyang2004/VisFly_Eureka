#!/usr/bin/env python3
"""Lightweight runner for VisFly-Eureka training and evaluation."""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch as th

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "VisFly"))

from VisFly.utils.common import load_yaml_config

from algorithms.BPTT_series.BPTT import BPTT
from algorithms.BPTT_series.SHAC import SHAC
from VisFly.utils.algorithms.PPO import PPO

ENV_SPECS = {
    "navigation": ("NavigationEnv.py", "NavigationEnv"),
    "hover": ("HoverEnv.py", "HoverEnv"),
    "object_tracking": ("ObjectTrackingEnv.py", "ObjectTrackingEnv"),
    "tracking": ("TrackingEnv.py", "TrackingEnv"),
    "catch": ("CatchEnv.py", "CatchEnv"),
    "landing": ("LandingEnv.py", "LandingEnv"),
    "racing": ("RacingEnv.py", "RacingEnv"),
    "flip": ("FlipEnv.py", "FlipEnv"),
    "vis_landing": ("VisLanding.py", "VisLandingEnv"),
    "circling": ("CirclingEnv.py", "CirclingEnv"),
}

ALGORITHM_REGISTRY = {
    "bptt": BPTT,
    "ppo": PPO,
    "shac": SHAC,
}

LOGGER = logging.getLogger(__name__)


@lru_cache(maxsize=None)
def resolve_env_class(name: str) -> Any:
    module_file, class_name = ENV_SPECS[name]
    module_path = PROJECT_ROOT / "envs" / module_file
    spec = importlib.util.spec_from_file_location(class_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load {class_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VisFly-Eureka experiments")
    parser.add_argument("--env", "-e", choices=sorted(ENV_SPECS.keys()), default="hover")
    parser.add_argument("--algorithm", "-a", choices=sorted(ALGORITHM_REGISTRY.keys()), default="bptt")
    parser.add_argument("--train", "-t", type=int, default=1, help="1=train, 0=test")
    parser.add_argument("--comment", "-c", type=str, default=None, help="Experiment note")
    parser.add_argument("--seed", "-s", type=int, default=42)
    parser.add_argument("--weight", "-w", type=str, default=None, help="Checkpoint to load")
    parser.add_argument("--reward_function_path", type=str, default=None)
    parser.add_argument("--num_eval_episodes", type=int, default=None)
    parser.add_argument("--video_fps", type=int, default=30)
    parser.add_argument("--learning_steps", type=int, default=None)
    parser.add_argument("--num_agents", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def load_configs(env_name: str, algorithm: str) -> tuple[Dict[str, Any], Dict[str, Any]]:
    config_root = PROJECT_ROOT / "configs"
    env_cfg_path = config_root / "envs" / f"{env_name}.yaml"
    alg_cfg_path = config_root / "algs" / env_name / f"{algorithm}.yaml"

    if not env_cfg_path.exists():
        raise FileNotFoundError(f"Missing environment config: {env_cfg_path}")
    if not alg_cfg_path.exists():
        raise FileNotFoundError(f"Missing algorithm config: {alg_cfg_path}")

    env_config = load_yaml_config(str(env_cfg_path))
    alg_config = load_yaml_config(str(alg_cfg_path))
    return env_config, alg_config


def apply_config_overrides(
    env_config: Dict[str, Any], alg_config: Dict[str, Any], args: argparse.Namespace
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    env_cfg = json.loads(json.dumps(env_config))  # simple deep copy
    alg_cfg = json.loads(json.dumps(alg_config))

    if "env_overrides" in alg_cfg:
        env_cfg.setdefault("env", {}).update(alg_cfg["env_overrides"])

    if args.learning_steps is not None:
        alg_cfg.setdefault("learn", {})["total_timesteps"] = args.learning_steps

    if args.num_agents is not None:
        env_cfg.setdefault("env", {})["num_agent_per_scene"] = args.num_agents

    if args.device is not None:
        env_cfg.setdefault("env", {})["device"] = args.device
        alg_cfg.setdefault("algorithm", {})["device"] = args.device

    comment = args.comment or f"{args.env}_{args.algorithm}"
    alg_cfg["comment"] = comment

    return env_cfg, alg_cfg


def get_env_class_registry() -> Dict[str, Any]:
    """Return env-name -> env-class registry (primarily for tests/mocking)."""
    return {name: resolve_env_class(name) for name in ENV_SPECS}


def create_environment(
    env_name: str,
    env_config: Dict[str, Any],
    training: Optional[bool] = None,
    *,
    is_training: Optional[bool] = None,
) -> Any:
    """
    Create an environment instance.

    Supports the legacy `is_training` keyword used by older tests.
    """
    training_mode = is_training if is_training is not None else bool(training)
    env_cls = get_env_class_registry()[env_name]
    key = "env" if training_mode else "eval_env"
    kwargs = dict(env_config[key])
    kwargs.pop("name", None)

    # Filter kwargs that are not accepted by the constructor; attach them afterwards.
    import inspect

    signature = inspect.signature(env_cls)
    accepted = set(signature.parameters.keys())
    filtered_out: Dict[str, Any] = {}
    ctor_kwargs: Dict[str, Any] = {}
    for k, v in kwargs.items():
        if k in accepted:
            ctor_kwargs[k] = v
        else:
            filtered_out[k] = v

    env = env_cls(**ctor_kwargs)
    for k, v in filtered_out.items():
        # Enforce compatibility: if tensor_output is False, requires_grad must be False
        if k == 'requires_grad' and not env.tensor_output and v:
            v = False
        setattr(env, k, v)
    return env


def inject_reward_function(reward_path: Optional[str], env_name: str) -> None:
    if not reward_path:
        return

    file_path = Path(reward_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Reward script not found: {reward_path}")

    env_cls = resolve_env_class(env_name)
    exec_globals: Dict[str, Any] = {"torch": th, "th": th, "np": np, "numpy": np}
    exec(file_path.read_text(), exec_globals)

    if "get_reward" not in exec_globals:
        raise RuntimeError("Reward script must define get_reward()")

    env_cls.get_reward = exec_globals["get_reward"]  # type: ignore[attr-defined]
    LOGGER.info("Injected custom reward from %s", reward_path)


def create_algorithm(
    algorithm: str, alg_config: Dict[str, Any], env: Any, save_dir: Path
) -> Any:
    algo_cls = ALGORITHM_REGISTRY[algorithm]
    algo_kwargs = dict(alg_config.get("algorithm", {}))
    comment = alg_config.get("comment")

    if algorithm == "ppo":
        return algo_cls(env=env, comment=comment, tensorboard_log=str(save_dir), **algo_kwargs)

    return algo_cls(env=env, comment=comment, save_path=str(save_dir), **algo_kwargs)


def maybe_resume_training(model: Any, args: argparse.Namespace, base_dir: Path, env: Any) -> Any:
    if not args.weight:
        return model

    weight_path = Path(args.weight)
    if not weight_path.is_absolute():
        weight_path = base_dir / weight_path

    LOGGER.info("Loading weights from %s", weight_path)
    model_cls = model.__class__
    return model_cls.load(str(weight_path), env=env)


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def reward_to_float(reward: Any) -> float:
    if isinstance(reward, dict):
        source = reward.get("reward", reward)
        return reward_to_float(source)
    if isinstance(reward, th.Tensor):
        return float(reward.float().mean().item())
    array = np.asarray(reward, dtype=np.float32)
    return float(array.mean())


def done_to_bool(done: Any) -> bool:
    if isinstance(done, th.Tensor):
        return bool(done.bool().all().item())
    array = np.asarray(done)
    return bool(array.all())


def snapshot_env(env: Any) -> Dict[str, np.ndarray]:
    def to_numpy(value: Any) -> np.ndarray:
        if isinstance(value, th.Tensor):
            return value.detach().cpu().numpy().copy()
        return np.array(value, copy=True)

    return {
        "position": to_numpy(env.position),
        "velocity": to_numpy(env.velocity),
        "orientation": to_numpy(env.orientation),
        "angular_velocity": to_numpy(env.angular_velocity),
        "target": to_numpy(getattr(env, "target", np.zeros(3))),
    }


# --- Backwards-compatible helpers expected by tests/unit/test_run_enhancements.py ---

def collect_trajectory_data(env: Any, obs: Any = None) -> Dict[str, np.ndarray]:
    """Legacy name for snapshotting environment state."""
    _ = obs
    return snapshot_env(env)


def create_trajectory_plot(
    trajectory: List[Dict[str, np.ndarray]], output_dir: Path, episode_idx: int = 0
) -> None:
    """Legacy name for trajectory plotting."""
    return plot_trajectory(trajectory, output_dir, episode_idx)


def record_video_frame(env: Any, video_writer: Any, video_params: Optional[Dict[str, Any]]) -> Any:
    """
    Legacy name for video recording that tolerates missing OpenCV.
    """
    if video_params is None:
        return video_writer
    if cv2 is None:
        return None

    # Older callers used 'init' sentinel.
    if video_writer == "init":
        video_writer = None

    path = Path(video_params["path"])
    fps = int(video_params.get("fps", 30))
    return record_frame(env, video_writer, path, fps)


def plot_trajectory(trajectory: List[Dict[str, np.ndarray]], output: Path, episode_idx: int) -> None:
    positions = np.stack([step["position"] for step in trajectory])
    velocities = np.stack([step["velocity"] for step in trajectory])

    if positions.ndim == 2:
        positions = positions[:, None, :]
    if velocities.ndim == 2:
        velocities = velocities[:, None, :]

    timesteps = np.arange(len(positions))
    num_agents = positions.shape[1]

    fig = plt.figure(figsize=(16, 12))

    ax1 = fig.add_subplot(2, 2, 1, projection="3d")
    for agent_idx in range(min(num_agents, 4)):
        x, y, z = positions[:, agent_idx, :].T
        ax1.plot(x, y, z, linewidth=1.5)
        ax1.scatter(x[0], y[0], z[0], color="green", marker="o", s=50)
        ax1.scatter(x[-1], y[-1], z[-1], color="red", marker="x", s=60)

    target = trajectory[0]["target"]
    target = target.reshape(-1, target.shape[-1])
    ax1.scatter(target[:, 0], target[:, 1], target[:, 2], color="blue", marker="*", s=120)
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_zlabel("Z (m)")
    ax1.set_title("3D Trajectory")

    ax2 = fig.add_subplot(2, 2, 2)
    mean_pos = positions.mean(axis=1)
    for idx, label in enumerate(["X", "Y", "Z"]):
        ax2.plot(timesteps, mean_pos[:, idx], label=label)
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Position (m)")
    ax2.legend()
    ax2.grid(True)

    ax3 = fig.add_subplot(2, 2, 3)
    mean_speed = np.linalg.norm(velocities.mean(axis=1), axis=1)
    ax3.plot(timesteps, mean_speed, color="purple")
    ax3.set_xlabel("Time Step")
    ax3.set_ylabel("Speed (m/s)")
    ax3.grid(True)

    ax4 = fig.add_subplot(2, 2, 4)
    target_pos = target[0]
    distances = np.linalg.norm(positions - target_pos, axis=2).mean(axis=1)
    ax4.plot(timesteps, distances, color="orange")
    ax4.set_xlabel("Time Step")
    ax4.set_ylabel("Distance (m)")
    ax4.grid(True)

    plt.tight_layout()
    fig_path = output / f"trajectory_episode_{episode_idx:03d}.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Saved trajectory plot: %s", fig_path)


def ensure_video_writer(writer: Optional[cv2.VideoWriter], frame: np.ndarray, path: Path, fps: int) -> cv2.VideoWriter:
    if writer is not None:
        return writer
    height, width = frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer at {path}")
    return writer


def record_frame(env: Any, writer: Optional[cv2.VideoWriter], path: Path, fps: int) -> cv2.VideoWriter:
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


def evaluate(args: argparse.Namespace) -> bool:
    LOGGER.info(
        "Starting testing: env=%s algorithm=%s weight=%s",
        args.env,
        args.algorithm,
        args.weight,
    )

    env_config, alg_config = load_configs(args.env, args.algorithm)
    env_config, alg_config = apply_config_overrides(env_config, alg_config, args)

    training_dir = ensure_directory(PROJECT_ROOT / "results" / "training" / args.env)
    testing_dir = ensure_directory(PROJECT_ROOT / "results" / "testing" / args.env)
    video_dir = ensure_directory(testing_dir / "videos")
    plot_dir = ensure_directory(testing_dir / "plots")

    test_env = create_environment(args.env, env_config, training=False)

    if not args.weight:
        raise ValueError("Testing mode requires --weight")

    weight_path = Path(args.weight)
    if not weight_path.is_absolute():
        weight_path = training_dir / weight_path

    model_cls = ALGORITHM_REGISTRY[args.algorithm]
    model = model_cls.load(str(weight_path), env=test_env)

    max_steps = env_config["eval_env"].get("max_episode_steps", 256)

    if args.num_eval_episodes is not None:
        num_episodes = args.num_eval_episodes
    elif "test" in alg_config and "episodes" in alg_config["test"]:
        num_episodes = alg_config["test"]["episodes"]
    elif "learn" in alg_config and "evaluation_episodes" in alg_config["learn"]:
        num_episodes = alg_config["learn"]["evaluation_episodes"]
    else:
        num_episodes = 10

    all_results: List[Dict[str, Any]] = []
    success_count = 0

    LOGGER.info("Running %d evaluation episodes...", num_episodes)

    for episode_idx in range(num_episodes):
        LOGGER.info("Episode %d/%d", episode_idx + 1, num_episodes)

        obs = test_env.reset()
        total_reward = 0.0
        steps = 0
        writer: Optional[cv2.VideoWriter] = None
        trajectory = [snapshot_env(test_env)]

        episode_video = video_dir / f"episode_{episode_idx:03d}.mp4"

        for _ in range(max_steps):
            action = model.predict(obs, deterministic=True)
            if isinstance(action, tuple):
                action = action[0]

            obs, reward, done, _ = test_env.step(action, is_test=True)
            total_reward += reward_to_float(reward)
            steps += 1

            trajectory.append(snapshot_env(test_env))
            writer = record_frame(test_env, writer, episode_video, args.video_fps)

            if done_to_bool(done):
                break

        if writer is not None:
            writer.release()
            LOGGER.info("Saved video: %s", episode_video)

        plot_trajectory(trajectory, plot_dir, episode_idx)

        success = bool(test_env.get_success().bool().any().item())
        success_count += int(success)

        LOGGER.info(
            "Episode %d: steps=%d reward=%.4f success=%s",
            episode_idx + 1,
            steps,
            total_reward,
            success,
        )

        all_results.append(
            {
                "episode": episode_idx,
                "steps": steps,
                "total_reward": total_reward,
                "average_reward": total_reward / max(steps, 1),
                "success": success,
            }
        )

    total_steps = sum(result["steps"] for result in all_results)
    total_reward = sum(result["total_reward"] for result in all_results)
    success_rate = success_count / num_episodes

    LOGGER.info("\n%s", "=" * 50)
    LOGGER.info("EVALUATION SUMMARY")
    LOGGER.info("Episodes: %d", num_episodes)
    LOGGER.info("Success Rate: %.2f%% (%d/%d)", success_rate * 100, success_count, num_episodes)
    LOGGER.info("Average Episode Reward: %.4f", total_reward / num_episodes)
    LOGGER.info("Average Episode Steps: %.1f", total_steps / num_episodes)
    LOGGER.info("%s\n", "=" * 50)

    results_path = testing_dir / f"test_results_{(args.comment or 'default')}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "environment": args.env,
                "algorithm": args.algorithm,
                "weight_file": str(weight_path),
                "num_episodes": num_episodes,
                "success_rate": success_rate,
                "success_count": success_count,
                "average_episode_reward": total_reward / num_episodes,
                "average_episode_steps": total_steps / num_episodes,
                "total_reward": total_reward,
                "total_steps": total_steps,
                "episodes": all_results,
            },
            handle,
            indent=2,
        )

    LOGGER.info("Results saved to: %s", results_path)
    LOGGER.info("Videos saved to: %s", video_dir)
    LOGGER.info("Trajectory plots saved to: %s", plot_dir)

    return True


def train(args: argparse.Namespace) -> bool:
    LOGGER.info(
        "Starting training: env=%s algorithm=%s comment=%s",
        args.env,
        args.algorithm,
        args.comment,
    )

    env_config, alg_config = load_configs(args.env, args.algorithm)
    env_config, alg_config = apply_config_overrides(env_config, alg_config, args)

    train_dir = ensure_directory(PROJECT_ROOT / "results" / "training" / args.env)
    train_env = create_environment(args.env, env_config, training=True)

    # BPTT requires tensor_output=True and requires_grad=True for analytical gradient RL
    # Other algorithms use numpy outputs
    if args.algorithm == "bptt" or args.algorithm == "shac":
        train_env.tensor_output = True
        train_env.requires_grad = True
    else:
        train_env.tensor_output = False
        train_env.requires_grad = False

    inject_reward_function(args.reward_function_path, args.env)

    model = create_algorithm(args.algorithm, alg_config, train_env, train_dir)
    model = maybe_resume_training(model, args, train_dir, train_env)

    learn_cfg = alg_config.get("learn", {})
    if args.algorithm == "ppo":
        total_timesteps = int(learn_cfg.get("total_timesteps", 0))
        model.learn(total_timesteps=total_timesteps)
    else:
        # Filter learn_cfg to only include parameters accepted by learn() method
        # BPTT.learn() accepts: total_timesteps, callback, log_interval, tb_log_name, reset_num_timesteps, progress_bar
        valid_learn_params = {'total_timesteps', 'callback', 'log_interval', 'tb_log_name', 'reset_num_timesteps', 'progress_bar'}
        filtered_learn_cfg = {k: v for k, v in learn_cfg.items() if k in valid_learn_params}
        model.learn(**filtered_learn_cfg)

    model.save()
    LOGGER.info("Training completed and model saved.")
    return True


def configure_logging() -> None:
    if logging.getLogger().hasHandlers():
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


def main() -> int:
    args = parse_args()
    configure_logging()
    th.manual_seed(args.seed)

    try:
        if args.train:
            success = train(args)
        else:
            success = evaluate(args)
        return 0 if success else 1
    except Exception as exc:  # pragma: no cover - surface helpful logs
        LOGGER.exception("Run failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

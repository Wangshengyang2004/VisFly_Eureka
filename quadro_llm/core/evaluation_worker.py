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
from typing import Any, Dict, List, Optional

import yaml
import torch

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - optional dependency for video export
    cv2 = None


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
    import numpy as np

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

    if frame.ndim == 3 and frame.shape[0] in (1, 3) and frame.shape[0] != frame.shape[-1]:
        # Channel-first -> channel-last
        frame = np.transpose(frame, (1, 2, 0))

    if frame.ndim == 2:
        frame = np.stack([frame] * 3, axis=-1)

    if frame.shape[-1] == 1:
        frame = np.repeat(frame, 3, axis=-1)

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
        
        # Track GPU memory if using CUDA
        peak_memory_mb = 0
        if torch.cuda.is_available() and alg_device.startswith("cuda"):
            gpu_id = int(alg_device.split(":")[1]) if ":" in alg_device else 0
            try:
                initial_memory = torch.cuda.memory_allocated(gpu_id) // (1024**2)  # MB
                torch.cuda.reset_peak_memory_stats(gpu_id)
                logger.info(f"Initial GPU memory: {initial_memory}MB")
            except Exception as e:
                logger.warning(f"GPU memory tracking failed: {e}")
        
        # Prepare training arguments
        train_args = {"total_timesteps": int(train_steps)}
        if log_interval is not None:
            train_args["log_interval"] = log_interval
        
        # Use getattr to avoid static type check issues
        getattr(model, "learn")(**train_args)
        training_time = time.time() - t0
        
        # Get peak GPU memory usage
        if torch.cuda.is_available() and alg_device.startswith("cuda"):
            gpu_id = int(alg_device.split(":")[1]) if ":" in alg_device else 0
            try:
                peak_memory_mb = torch.cuda.max_memory_allocated(gpu_id) // (1024**2)  # MB
                final_memory = torch.cuda.memory_allocated(gpu_id) // (1024**2)  # MB
                logger.info(f"Peak GPU memory: {peak_memory_mb}MB, Final: {final_memory}MB")
            except Exception as e:
                logger.warning(f"GPU memory tracking failed: {e}")
        
        logger.info(f"Training completed in {training_time:.2f}s")
        
        # Save trained model
        model_saved = False
        model_save_path = Path(output_dir) / "trained_model.zip"
        try:
            if hasattr(model, 'save'):
                model.save(str(model_save_path))
                model_saved = True
                logger.info(f"Saved trained model to: {model_save_path}")
            else:
                logger.warning("Model does not support saving (no 'save' method)")
        except Exception as e:
            logger.warning(f"Failed to save model: {e}")

        # Create evaluation env (requires_grad False)
        eval_env_config = {**eval_env_config, "requires_grad": False}
        logger.info("[stage] constructing eval env ...")
        env_eval = EnvClass(**eval_env_config)
        logger.info("[stage] eval env constructed (inherits injected reward from class)")

        # Run evaluation episodes with detailed statistics
        import numpy as np
        import torch as th

        requested_eval_episodes = int(eval_episodes)
        eval_runs = max(10, requested_eval_episodes)
        logger.info(
            f"[stage] evaluating trained policy over {eval_runs} episodes (requested: {requested_eval_episodes})"
        )

        success_flags: List[bool] = []
        collision_flags: List[bool] = []
        episode_lengths: List[float] = []
        episode_rewards: List[float] = []
        final_distances: List[float] = []
        episode_details: List[Dict[str, Any]] = []
        saved_videos: List[str] = []

        max_steps = int(eval_env_config["max_episode_steps"])
        video_enabled = bool(eval_env_config.get("visual", False))
        video_fps = float(eval_env_config.get("video_fps", 30.0))
        video_dir = Path(output_dir) / "videos"
        if video_enabled:
            video_dir.mkdir(parents=True, exist_ok=True)
            if cv2 is None:
                logger.warning(
                    "OpenCV is unavailable; disabling video export despite visual evaluation."
                )
                video_enabled = False

        model.policy.set_training_mode(False) if hasattr(model, "policy") else None

        for episode_idx in range(eval_runs):
            obs = env_eval.reset()
            done_flag = False
            step_count = 0
            episode_reward = 0.0
            last_info = None
            video_writer = None
            frames_captured = 0
            video_path = None

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

                if isinstance(done_arr, (np.ndarray, list)):
                    done_flag = bool(np.any(done_arr))
                elif isinstance(done_arr, th.Tensor):
                    done_flag = bool(done_arr.any().item())
                else:
                    done_flag = bool(done_arr)

                if video_enabled:
                    frame = None
                    try:
                        frame = env_eval.render()
                    except Exception as render_err:  # pragma: no cover - render depends on runtime
                        logger.debug(f"Episode {episode_idx}: render failed ({render_err})")
                        frame = None
                    rgb_frame = _extract_rgb_frame(frame)
                    if rgb_frame is not None and cv2 is not None:
                        if video_writer is None:
                            height, width = rgb_frame.shape[:2]
                            fourcc = cv2.VideoWriter_fourcc(*"avc1")  # Use avc1 for better compatibility
                            video_path = video_dir / f"episode_{episode_idx:02d}.mp4"
                            video_writer = cv2.VideoWriter(
                                str(video_path), fourcc, video_fps, (width, height)
                            )
                            if not video_writer.isOpened():
                                logger.warning(
                                    f"Failed to open AVC1 video writer for {video_path}. Disabling video capture."
                                )
                                video_writer.release()
                                video_writer = None
                                video_path = None
                                video_enabled = False
                        if video_writer is not None:
                            bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
                            video_writer.write(bgr_frame)
                            frames_captured += 1

                step_count += 1

            if video_writer is not None:
                video_writer.release()
                if video_path is not None:
                    saved_videos.append(str(video_path))

            if hasattr(env_eval, "get_success"):
                success_tensor = env_eval.get_success()
                if isinstance(success_tensor, th.Tensor):
                    success_bool = bool(success_tensor.any().item())
                else:
                    success_bool = bool(success_tensor)
            elif last_info:
                success_bool = any(
                    info.get("is_success", False) for info in last_info if isinstance(info, dict)
                )
            else:
                success_bool = episode_reward > 0

            if not success_bool and last_info:
                success_bool = any(
                    info.get("is_success", False) for info in last_info if isinstance(info, dict)
                )

            success_flags.append(success_bool)

            collision_flag = False
            if hasattr(env_eval, "is_collision"):
                try:
                    collision_flag = bool(env_eval.is_collision.any().item())
                except Exception:
                    collision_flag = bool(env_eval.is_collision)
            collision_flags.append(collision_flag)

            episode_lengths.append(float(step_count))
            episode_rewards.append(float(episode_reward))

            final_distance = None
            if hasattr(env_eval, "position") and hasattr(env_eval, "target"):
                try:
                    distance_tensor = (env_eval.position - env_eval.target).norm(dim=1)
                    final_distance = float(distance_tensor.mean().item())
                    final_distances.append(final_distance)
                except Exception:
                    final_distance = None

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
                "frames_captured": frames_captured,
            }
            if final_distance is not None:
                episode_record["final_distance_to_target"] = final_distance
            if video_path is not None:
                episode_record["video_path"] = str(video_path)

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
            "model_saved": model_saved,
            "model_path": str(model_save_path) if model_saved else None,
            "evaluation_runs": len(success_flags),
            "aggregate_statistics": aggregate_statistics,
            "episode_statistics": episode_details,
            "video_paths": saved_videos,
        }

        stats_payload = {
            "identifier": identifier,
            "aggregate_statistics": aggregate_statistics,
            "episode_statistics": episode_details,
            "video_paths": saved_videos,
        }

        stats_file = Path(output_dir) / "evaluation_stats.json"
        with open(stats_file, "w") as stats_f:
            json.dump(stats_payload, stats_f, indent=2)

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

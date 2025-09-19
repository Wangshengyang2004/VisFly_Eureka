"""
Tensorboard Log Parsing for VisFly Training Results

This module provides functionality to parse tensorboard logs from VisFly training
and generate detailed feedback like the real Eureka system.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging


def load_tensorboard_logs(logdir: str) -> Dict[str, List[float]]:
    """
    Load tensorboard logs from directory (similar to real Eureka's function).

    Args:
        logdir: Path to tensorboard log directory

    Returns:
        Dictionary mapping metric names to value lists
    """
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )

        # Initialize accumulator
        event_acc = EventAccumulator(logdir)
        event_acc.Reload()

        # Extract scalar metrics
        logs = {}
        scalar_tags = event_acc.Tags()["scalars"]

        for tag in scalar_tags:
            scalar_events = event_acc.Scalars(tag)
            values = [event.value for event in scalar_events]
            logs[tag] = values

        return logs

    except ImportError:
        logging.warning("tensorboard not installed, using CSV fallback")
        return load_csv_logs(logdir)
    except Exception as e:
        logging.error(f"Failed to load tensorboard logs: {e}")
        return {}


def load_csv_logs(logdir: str) -> Dict[str, List[float]]:
    """
    Fallback: Load training logs from CSV files in VisFly format.

    Args:
        logdir: Path to log directory containing CSV files

    Returns:
        Dictionary mapping metric names to value lists
    """
    logs = {}
    log_path = Path(logdir)

    # Look for common VisFly log files
    csv_files = list(log_path.glob("*.csv"))

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)

            # Extract metrics from dataframe
            for column in df.columns:
                if column.lower() in ["step", "time", "timestamp"]:
                    continue

                values = df[column].dropna().tolist()
                if values:
                    logs[column] = values

        except Exception as e:
            logging.warning(f"Failed to load CSV {csv_file}: {e}")

    # If no CSV files, try to extract from stdout logs
    if not logs:
        logs = extract_logs_from_stdout(logdir)

    return logs


def extract_logs_from_stdout(logdir: str) -> Dict[str, List[float]]:
    """
    Extract training metrics from stdout log files.

    Args:
        logdir: Directory containing log files

    Returns:
        Dictionary of extracted metrics
    """
    logs = {}
    log_path = Path(logdir)

    # Look for stdout log files
    log_files = list(log_path.glob("*.log")) + list(log_path.glob("stdout.txt"))

    for log_file in log_files:
        try:
            with open(log_file, "r") as f:
                content = f.read()

            # Extract common VisFly BPTT metrics
            metrics = {
                "ep_rew_mean": extract_metric_from_text(
                    content, r"rollout/ep_rew_mean\s*\|\s*([-\d\.]+)"
                ),
                "ep_len_mean": extract_metric_from_text(
                    content, r"rollout/ep_len_mean\s*\|\s*([-\d\.]+)"
                ),
                "success_rate": extract_metric_from_text(
                    content, r"rollout/success_rate\s*\|\s*([-\d\.]+)"
                ),
                "actor_loss": extract_metric_from_text(
                    content, r"train/actor_loss\s*\|\s*([-\d\.]+)"
                ),
                "critic_loss": extract_metric_from_text(
                    content, r"train/critic_loss\s*\|\s*([-\d\.]+)"
                ),
            }

            # Add non-empty metrics to logs
            for key, values in metrics.items():
                if values:
                    logs[key] = values

        except Exception as e:
            logging.warning(f"Failed to parse log file {log_file}: {e}")

    return logs


def extract_metric_from_text(text: str, pattern: str) -> List[float]:
    """Extract metric values from text using regex pattern."""
    import re

    matches = re.findall(pattern, text)

    try:
        return [float(match) for match in matches]
    except ValueError:
        return []


def generate_eureka_style_feedback(
    best_logs: Dict[str, List[float]], baseline_logs: Dict[str, List[float]] = None
) -> str:
    """
    Generate simple feedback like real Eureka does.

    Args:
        best_logs: Tensorboard logs from best reward function
        baseline_logs: Tensorboard logs from baseline (optional)

    Returns:
        Formatted feedback string with training statistics
    """
    if not best_logs:
        return "No training logs available for analysis."

    feedback_parts = []

    # Calculate epoch frequency for ~10 data points (like real Eureka)
    max_iterations = len(list(best_logs.values())[0]) if best_logs else 100
    epoch_freq = max(int(max_iterations // 10), 1)

    # Add header like real Eureka
    feedback_parts.append(
        f"We trained a RL policy using the provided reward function code and tracked the values of the individual components "
        f"in the reward function as well as global policy metrics such as success rates and episode lengths "
        f"after every {epoch_freq} epochs and the maximum, mean, minimum values encountered:\n"
    )

    # Process each metric exactly like real Eureka (lines 250-267)
    for metric_name, values in best_logs.items():
        if not values or len(values) == 0 or "/" in metric_name:
            continue

        # Truncate to ~10 points like real Eureka
        metric_cur = ["{:.2f}".format(x) for x in values[::epoch_freq]]
        metric_cur_max = max(values)
        metric_cur_mean = sum(values) / len(values)
        metric_cur_min = min(values)

        # Use display name like real Eureka
        if metric_name == "ep_rew_mean" or metric_name == "rollout/ep_rew_mean":
            display_name = "episode_reward"
        elif metric_name == "success_rate" or metric_name == "rollout/success_rate":
            display_name = "task_score"
        elif metric_name == "ep_len_mean" or metric_name == "rollout/ep_len_mean":
            display_name = "episode_length"
        else:
            display_name = metric_name

        feedback_parts.append(
            f"{display_name}: {metric_cur}, Max: {metric_cur_max:.2f}, Mean: {metric_cur_mean:.2f}, Min: {metric_cur_min:.2f}"
        )

    return "\n".join(feedback_parts)


def get_display_metric_name(metric_name: str) -> str:
    """Convert internal metric names to display names."""
    name_mapping = {
        "ep_rew_mean": "episode_reward",
        "ep_len_mean": "episode_length",
        "success_rate": "task_score",
        "actor_loss": "actor_loss",
        "critic_loss": "critic_loss",
        "rollout/ep_rew_mean": "episode_reward",
        "rollout/ep_len_mean": "episode_length",
        "rollout/success_rate": "task_score",
        "train/actor_loss": "actor_loss",
        "train/critic_loss": "critic_loss",
        "consecutive_successes": "task_score",
    }

    return name_mapping.get(metric_name, metric_name)


def compute_reward_correlation(
    baseline_logs: Dict[str, List[float]], gpt_logs: Dict[str, List[float]]
) -> float:
    """
    Compute correlation between baseline (gt) and GPT reward like real Eureka.

    Args:
        baseline_logs: Baseline training logs
        gpt_logs: GPT reward training logs

    Returns:
        Correlation coefficient between reward curves
    """
    try:
        # Extract reward curves
        baseline_rewards = baseline_logs.get(
            "ep_rew_mean", baseline_logs.get("rollout/ep_rew_mean", [])
        )
        gpt_rewards = gpt_logs.get(
            "ep_rew_mean", gpt_logs.get("rollout/ep_rew_mean", [])
        )

        if not baseline_rewards or not gpt_rewards:
            return -1.0  # No correlation data available

        # Align lengths by taking minimum
        min_len = min(len(baseline_rewards), len(gpt_rewards))
        if min_len < 2:
            return -1.0

        baseline_rewards = baseline_rewards[:min_len]
        gpt_rewards = gpt_rewards[:min_len]

        # Compute correlation
        correlation = np.corrcoef(baseline_rewards, gpt_rewards)[0, 1]

        return correlation if not np.isnan(correlation) else -1.0

    except Exception as e:
        logging.warning(f"Failed to compute reward correlation: {e}")
        return -1.0


def find_tensorboard_logdir(job_output_dir: str) -> Optional[str]:
    """
    Find tensorboard log directory from job output.

    Args:
        job_output_dir: Job output directory

    Returns:
        Path to tensorboard logs or None
    """
    job_path = Path(job_output_dir)

    # First, look for event files directly (most reliable)
    event_files = list(job_path.glob("**/events.out.tfevents.*"))
    if event_files:
        return str(event_files[0].parent)

    # Common tensorboard directory patterns (fallback)
    patterns = [
        "**/*_1",  # VisFly/stable-baselines3 pattern (prioritize nested dirs)
        "**/logs", 
        "**/tb_logs",
        "**/runs",
        "**/tensorboard",  # Check this last since it might be parent dir
    ]

    for pattern in patterns:
        matches = list(job_path.glob(pattern))
        # Check if any matches actually contain event files
        for match in matches:
            if list(Path(match).glob("events.out.tfevents.*")):
                return str(match)

    return None


def create_tensorboard_dataframe(logs: Dict[str, List[float]]) -> pd.DataFrame:
    """
    Create a pandas DataFrame from TensorBoard logs for next iteration feedback.
    
    Args:
        logs: Dictionary mapping metric names to value lists
        
    Returns:
        DataFrame with training metrics over time
    """
    if not logs:
        return pd.DataFrame()
    
    # Find the maximum length to align all metrics
    max_length = max(len(values) for values in logs.values()) if logs else 0
    
    # Create DataFrame with step column
    df_data = {"step": list(range(max_length))}
    
    # Add each metric, padding with NaN if necessary
    for metric_name, values in logs.items():
        if values:
            # Pad values to match max length
            padded_values = values + [np.nan] * (max_length - len(values))
            df_data[metric_name] = padded_values
    
    return pd.DataFrame(df_data)


def sample_dataframe_uniformly(df: pd.DataFrame, n_samples: int = 10) -> pd.DataFrame:
    """
    Sample DataFrame uniformly to reduce size while preserving training progression.
    
    Args:
        df: Original DataFrame
        n_samples: Number of samples to extract (default 10)
        
    Returns:
        Sampled DataFrame with uniform intervals
    """
    if df.empty or len(df) <= n_samples:
        return df
    
    # Create uniform indices
    indices = np.linspace(0, len(df) - 1, n_samples, dtype=int)
    return df.iloc[indices].copy()


def append_dataframe_to_feedback(
    feedback: str, 
    tensorboard_logs: Dict[str, List[float]],
    selected_index: int = None,
    total_candidates: int = None
) -> str:
    """
    Append TensorBoard data as DataFrame to feedback for next iteration agent.
    Includes uniformly sampled data points to keep feedback concise.
    
    Args:
        feedback: Existing feedback string
        tensorboard_logs: TensorBoard logs dictionary
        
    Returns:
        Enhanced feedback with DataFrame summary and sampled data
    """
    if not tensorboard_logs:
        return feedback
    
    df = create_tensorboard_dataframe(tensorboard_logs)
    if df.empty:
        return feedback
    
    # Sample DataFrame to keep feedback manageable
    sampled_df = sample_dataframe_uniformly(df, n_samples=10)
    
    # Add DataFrame summary to feedback
    enhanced_feedback = feedback + "\n\n"
    
    # Add reward function selection info if provided
    if selected_index is not None:
        if total_candidates is not None:
            enhanced_feedback += f"## SELECTED REWARD FUNCTION: #{selected_index} (from {total_candidates} candidates)\n"
        else:
            enhanced_feedback += f"## SELECTED REWARD FUNCTION: #{selected_index}\n"
        enhanced_feedback += "The following TensorBoard data corresponds to this specific reward function.\n\n"
    
    enhanced_feedback += "## TensorBoard Training Data Summary\n"
    enhanced_feedback += f"Original dataset shape: {df.shape}, Sampled: {sampled_df.shape}\n"
    display_metrics = [get_display_metric_name(col) for col in df.columns]
    enhanced_feedback += f"Available metrics: {display_metrics}\n\n"
    
    # Add key statistics from full data
    enhanced_feedback += "### Key Statistics (Full Training):\n"
    for column in df.columns:
        if column != "step" and not df[column].isna().all():
            values = df[column].dropna()
            if len(values) > 0:
                display_name = get_display_metric_name(column)
                enhanced_feedback += f"- {display_name}: mean={values.mean():.3f}, std={values.std():.3f}, "
                enhanced_feedback += f"min={values.min():.3f}, max={values.max():.3f}\n"
    
    # Add sampled data for detailed analysis
    enhanced_feedback += "\n### Sampled Training Progression (10 points):\n"
    enhanced_feedback += "```\n"
    enhanced_feedback += sampled_df.to_string(float_format='%.3f', max_cols=8)
    enhanced_feedback += "\n```\n"
    
    # Add trend information
    enhanced_feedback += "\n### Training Trends:\n"
    for column in ["ep_rew_mean", "success_rate", "rollout/ep_rew_mean", "rollout/success_rate", "consecutive_successes", "task_score"]:
        if column in df.columns and not df[column].isna().all():
            values = df[column].dropna()
            if len(values) > 1:
                trend = "increasing" if values.iloc[-1] > values.iloc[0] else "decreasing"
                display_name = get_display_metric_name(column)
                enhanced_feedback += (
                    f"- {display_name}: {trend} trend (start: {values.iloc[0]:.3f}, "
                    f"end: {values.iloc[-1]:.3f})\n"
                )
    
    return enhanced_feedback


def extract_success_metric(logs: Dict[str, List[float]]) -> float:
    """Extract the main success metric for ranking."""
    # Priority order for success metrics
    success_keys = [
        "task_score",
        "success_rate",
        "consecutive_successes",
        "rollout/success_rate",
        "ep_rew_mean",
        "rollout/ep_rew_mean",
    ]

    for key in success_keys:
        if key in logs and logs[key]:
            return max(logs[key])  # Return max value achieved

    return -10000.0  # DUMMY_FAILURE like real Eureka

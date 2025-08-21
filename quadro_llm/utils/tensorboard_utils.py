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
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        
        # Initialize accumulator
        event_acc = EventAccumulator(logdir)
        event_acc.Reload()
        
        # Extract scalar metrics
        logs = {}
        scalar_tags = event_acc.Tags()['scalars']
        
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
                if column.lower() in ['step', 'time', 'timestamp']:
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
            with open(log_file, 'r') as f:
                content = f.read()
                
            # Extract common VisFly BPTT metrics
            metrics = {
                'ep_rew_mean': extract_metric_from_text(content, r'rollout/ep_rew_mean\s*\|\s*([-\d\.]+)'),
                'ep_len_mean': extract_metric_from_text(content, r'rollout/ep_len_mean\s*\|\s*([-\d\.]+)'),
                'success_rate': extract_metric_from_text(content, r'rollout/success_rate\s*\|\s*([-\d\.]+)'),
                'actor_loss': extract_metric_from_text(content, r'train/actor_loss\s*\|\s*([-\d\.]+)'),
                'critic_loss': extract_metric_from_text(content, r'train/critic_loss\s*\|\s*([-\d\.]+)'),
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
    best_logs: Dict[str, List[float]], 
    baseline_logs: Dict[str, List[float]] = None
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
        metric_cur = ['{:.2f}'.format(x) for x in values[::epoch_freq]]
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
        'ep_rew_mean': 'episode_reward',
        'ep_len_mean': 'episode_length', 
        'success_rate': 'consecutive_successes',
        'actor_loss': 'actor_loss',
        'critic_loss': 'critic_loss',
        'rollout/ep_rew_mean': 'episode_reward',
        'rollout/ep_len_mean': 'episode_length',
        'rollout/success_rate': 'consecutive_successes',
        'train/actor_loss': 'actor_loss',
        'train/critic_loss': 'critic_loss',
    }
    
    return name_mapping.get(metric_name, metric_name)


def compute_reward_correlation(
    baseline_logs: Dict[str, List[float]], 
    gpt_logs: Dict[str, List[float]]
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
        baseline_rewards = baseline_logs.get('ep_rew_mean', baseline_logs.get('rollout/ep_rew_mean', []))
        gpt_rewards = gpt_logs.get('ep_rew_mean', gpt_logs.get('rollout/ep_rew_mean', []))
        
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
    
    # Common tensorboard directory patterns
    patterns = [
        "**/logs",
        "**/tensorboard", 
        "**/tb_logs",
        "**/runs",
        "**/*_1",  # VisFly/stable-baselines3 pattern
    ]
    
    for pattern in patterns:
        matches = list(job_path.glob(pattern))
        if matches:
            return str(matches[0])
    
    # Look for event files directly
    event_files = list(job_path.glob("**/events.out.tfevents.*"))
    if event_files:
        return str(event_files[0].parent)
    
    return None


def extract_success_metric(logs: Dict[str, List[float]]) -> float:
    """Extract the main success metric for ranking."""
    # Priority order for success metrics
    success_keys = [
        'consecutive_successes',
        'success_rate', 
        'rollout/success_rate',
        'ep_rew_mean',
        'rollout/ep_rew_mean'
    ]
    
    for key in success_keys:
        if key in logs and logs[key]:
            return max(logs[key])  # Return max value achieved
    
    return -10000.0  # DUMMY_FAILURE like real Eureka
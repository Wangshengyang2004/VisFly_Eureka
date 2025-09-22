#!/usr/bin/env python3
"""
TensorBoard Monitor for VisFly-Eureka Training

This script helps you find and monitor TensorBoard logs from your training runs.
"""

import argparse
import logging
import re
import subprocess
from pathlib import Path

from quadro_llm.utils.tensorboard_utils import find_tensorboard_logdir

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

TIMESTAMP_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}")


def find_all_tensorboard_logs(results_dir="results"):
    """Find all TensorBoard log directories in results."""
    results_path = Path(results_dir)
    if not results_path.exists():
        logger.error(f"Results directory not found: {results_dir}")
        return []
    
    tensorboard_dirs = []
    
    # Search for tensorboard directories
    patterns = [
        "**/tensorboard",
        "**/tb_logs", 
        "**/logs",
        "**/events.out.tfevents.*"
    ]
    
    for pattern in patterns:
        matches = list(results_path.glob(pattern))
        for match in matches:
            if match.is_file() and "events.out.tfevents" in match.name:
                # It's an event file, use parent directory
                tensorboard_dirs.append(str(match.parent))
            elif match.is_dir():
                # It's a directory that might contain tensorboard logs
                event_files = list(match.glob("**/events.out.tfevents.*"))
                if event_files:
                    tensorboard_dirs.append(str(match))

    return list(set(tensorboard_dirs))  # Remove duplicates


def get_tensorboard_run_choices(results_dir: str = "results"):
    """Condense TensorBoard directories down to top-level run choices."""
    results_path = Path(results_dir)
    if not results_path.exists():
        return []

    timestamp_runs = []
    other_runs = []

    for candidate in results_path.iterdir():
        if not candidate.is_dir():
            continue

        logdir = find_tensorboard_logdir(str(candidate))
        if not logdir:
            continue

        entry = (candidate.name, logdir)
        if TIMESTAMP_PATTERN.fullmatch(candidate.name):
            timestamp_runs.append(entry)
        else:
            other_runs.append(entry)

    if timestamp_runs or other_runs:
        timestamp_runs.sort(key=lambda x: x[0], reverse=True)
        other_runs.sort(key=lambda x: x[0])
        return timestamp_runs + other_runs

    # Fallback to legacy deep search when no top-level runs are found
    legacy_dirs = find_all_tensorboard_logs(results_dir)
    fallback_entries = []
    for logdir in sorted(legacy_dirs):
        path = Path(logdir)
        try:
            display = str(path.relative_to(results_path))
        except ValueError:
            display = path.name
        fallback_entries.append((display, str(path)))
    return fallback_entries


def find_latest_run_tensorboard():
    """Find TensorBoard logs for the most recent run."""
    results_path = Path("results")
    if not results_path.exists():
        return None

    # Get most recent directory
    run_dirs = [
        d
        for d in results_path.iterdir()
        if d.is_dir() and TIMESTAMP_PATTERN.fullmatch(d.name)
    ]
    if not run_dirs:
        return None
    
    latest_run = sorted(run_dirs, key=lambda x: x.name)[-1]
    logger.info(f"Latest run: {latest_run.name}")
    
    # Search for tensorboard logs in this run
    tensorboard_dir = find_tensorboard_logdir(str(latest_run))
    return tensorboard_dir


def launch_tensorboard(logdir, port=6006):
    """Launch TensorBoard server."""
    if not Path(logdir).exists():
        logger.error(f"Log directory not found: {logdir}")
        return False
    
    logger.info(f"Launching TensorBoard for: {logdir}")
    logger.info(f"TensorBoard will be available at: http://localhost:{port}")
    
    try:
        # Launch tensorboard
        cmd = ["tensorboard", "--logdir", logdir, "--port", str(port), "--host", "0.0.0.0"]
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to launch TensorBoard: {e}")
        return False
    except FileNotFoundError:
        logger.error("TensorBoard not found. Install with: pip install tensorboard")
        return False


def list_available_runs():
    """List all available training runs with TensorBoard logs."""
    logger.info("Searching for TensorBoard logs...")

    run_choices = get_tensorboard_run_choices()

    if not run_choices:
        logger.warning("No TensorBoard logs found!")
        logger.info("TensorBoard logs should be in:")
        logger.info("  - results/YYYY-MM-DD_HH-MM-SS/train/iter*/sample*/tensorboard/")
        logger.info("  - results/YYYY-MM-DD_HH-MM-SS/train/iter*/sample*/")
        return []

    logger.info(f"Found {len(run_choices)} TensorBoard runs:")
    for i, (display, _) in enumerate(run_choices, 1):
        logger.info(f"  {i}. {display}")

    return [logdir for _, logdir in run_choices]


def extract_training_metrics(results_dir=None):
    """Extract training metrics from the most recent run."""
    if results_dir is None:
        tensorboard_dir = find_latest_run_tensorboard()
    else:
        tensorboard_dir = find_tensorboard_logdir(results_dir)
    
    if not tensorboard_dir:
        logger.error("No TensorBoard logs found for metrics extraction")
        return
    
    # Use the tensorboard utils to load and display metrics
    from quadro_llm.utils.tensorboard_utils import load_tensorboard_logs, generate_eureka_style_feedback
    
    logger.info(f"Loading metrics from: {tensorboard_dir}")
    logs = load_tensorboard_logs(tensorboard_dir)
    
    if not logs:
        logger.warning("No metrics found in TensorBoard logs")
        return
    
    # Generate and display feedback
    feedback = generate_eureka_style_feedback(logs)
    print("\n" + "="*60)
    print("TRAINING METRICS SUMMARY")
    print("="*60)
    print(feedback)
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Monitor TensorBoard logs for VisFly-Eureka")
    parser.add_argument("--launch", "-l", action="store_true", help="Launch TensorBoard server")
    parser.add_argument("--port", "-p", type=int, default=6006, help="TensorBoard port (default: 6006)")
    parser.add_argument("--logdir", "-d", help="Specific log directory to use")
    parser.add_argument("--latest", action="store_true", help="Use latest run automatically")
    parser.add_argument("--list", action="store_true", help="List all available TensorBoard logs")
    parser.add_argument("--metrics", "-m", action="store_true", help="Extract and display training metrics")
    parser.add_argument("--results-dir", help="Specific results directory to analyze")
    
    args = parser.parse_args()
    
    if args.list:
        list_available_runs()
        return
    
    if args.metrics:
        extract_training_metrics(args.results_dir)
        return
    
    # Determine log directory
    logdir = None
    
    if args.logdir:
        logdir = args.logdir
    elif args.latest:
        logdir = find_latest_run_tensorboard()
        if not logdir:
            logger.error("No TensorBoard logs found in latest run")
            return
    else:
        # Interactive selection
        run_choices = get_tensorboard_run_choices()
        if not run_choices:
            logger.error("No TensorBoard logs found. Use --list to see search locations.")
            return
        elif len(run_choices) == 1:
            display, logdir = run_choices[0]
            logger.info(f"Using only available run: {display}")
        else:
            logger.info("Multiple TensorBoard runs found:")
            for i, (display, _) in enumerate(run_choices, 1):
                logger.info(f"  {i}. {display}")
            
            try:
                choice = int(input("Select directory (number): ")) - 1
                if 0 <= choice < len(run_choices):
                    display, logdir = run_choices[choice]
                    logger.info(f"Selected run: {display}")
                else:
                    logger.error("Invalid choice")
                    return
            except (ValueError, KeyboardInterrupt):
                logger.info("Selection cancelled")
                return
    
    if args.launch and logdir:
        launch_tensorboard(logdir, args.port)
    elif logdir:
        logger.info(f"TensorBoard log directory: {logdir}")
        logger.info(f"To launch TensorBoard, run:")
        logger.info(f"  tensorboard --logdir {logdir} --port {args.port}")
    else:
        logger.error("No log directory specified or found")


if __name__ == "__main__":
    main()

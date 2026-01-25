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

        entry = {
            "display": candidate.name,
            "run_root": str(candidate),
            "event_logdir": logdir,
        }
        if TIMESTAMP_PATTERN.fullmatch(candidate.name):
            timestamp_runs.append(entry)
        else:
            other_runs.append(entry)

    if timestamp_runs or other_runs:
        timestamp_runs.sort(key=lambda run: run["display"], reverse=True)
        other_runs.sort(key=lambda run: run["display"])
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
        fallback_entries.append(
            {
                "display": display,
                "run_root": str(path),
                "event_logdir": str(path),
            }
        )
    return fallback_entries


def find_latest_run_tensorboard():
    """Find TensorBoard logs for the most recent run."""
    run_choices = get_tensorboard_run_choices()
    if not run_choices:
        return None

    # run_choices already sorted with newest timestamps first
    latest = run_choices[0]
    logger.info(f"Latest run: {latest['display']}")
    return latest["run_root"]


def launch_tensorboard(logdir, port=60018):
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
    for i, run in enumerate(run_choices, 1):
        logger.info(f"  {i}. {run['display']}")

    return [run["run_root"] for run in run_choices]


def extract_training_metrics(results_dir=None):
    """Extract training metrics from the most recent run."""
    if results_dir is None:
        run_root = find_latest_run_tensorboard()
        tensorboard_dir = find_tensorboard_logdir(run_root) if run_root else None
    else:
        run_root = results_dir
        tensorboard_dir = find_tensorboard_logdir(results_dir)

    if not tensorboard_dir:
        logger.error("No TensorBoard logs found for metrics extraction")
        return

    # Use the tensorboard utils to load and display metrics
    from quadro_llm.utils.tensorboard_utils import load_tensorboard_logs, generate_eureka_style_feedback
    
    if results_dir is None and run_root:
        logger.info(f"Inspecting run directory: {run_root}")
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
    parser.add_argument("--port", "-p", type=int, default=60018, help="TensorBoard port (default: 60018)")
    parser.add_argument("--logdir", "-d", help="Specific log directory to use")
    parser.add_argument("--latest", action="store_true", help="Use latest run automatically")
    parser.add_argument("--list", action="store_true", help="List all available TensorBoard logs")
    parser.add_argument("--metrics", "-m", action="store_true", help="Extract and display training metrics")
    parser.add_argument("--results-dir", help="Specific results directory to analyze")
    parser.add_argument(
        "--no-launch",
        action="store_true",
        help="Only display the resolved directories; do not start TensorBoard",
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_available_runs()
        return
    
    if args.metrics:
        extract_training_metrics(args.results_dir)
        return
    
    # Determine run directory and representative logdir
    run_root = None
    event_logdir = None

    selected_display = None

    if args.logdir:
        run_root = args.logdir
        selected_display = Path(run_root).name
        event_logdir = find_tensorboard_logdir(run_root) or run_root
    elif args.latest:
        run_root = find_latest_run_tensorboard()
        if not run_root:
            logger.error("No TensorBoard logs found in latest run")
            return
        selected_display = Path(run_root).name
        event_logdir = find_tensorboard_logdir(run_root) or run_root
    else:
        # Interactive selection
        run_choices = get_tensorboard_run_choices()
        if not run_choices:
            logger.error("No TensorBoard logs found. Use --list to see search locations.")
            return
        elif len(run_choices) == 1:
            selection = run_choices[0]
            run_root = selection["run_root"]
            event_logdir = selection["event_logdir"]
            selected_display = selection["display"]
            logger.info(f"Using only available run: {selected_display}")
        else:
            logger.info("Multiple TensorBoard runs found:")
            for i, run in enumerate(run_choices, 1):
                logger.info(f"  {i}. {run['display']}")
            
            try:
                choice = int(input("Select directory (number): ")) - 1
                if 0 <= choice < len(run_choices):
                    selection = run_choices[choice]
                    run_root = selection["run_root"]
                    event_logdir = selection["event_logdir"]
                    selected_display = selection["display"]
                    logger.info(f"Selected run: {selected_display}")
                else:
                    logger.error("Invalid choice")
                    return
            except (ValueError, KeyboardInterrupt):
                logger.info("Selection cancelled")
                return

    if run_root:
        if selected_display:
            logger.info(f"TensorBoard run: {selected_display}")
        logger.info(f"TensorBoard run directory: {run_root}")
        if event_logdir and event_logdir != run_root:
            logger.info(f"First detected event logs: {event_logdir}")
        if args.no_launch:
            logger.info("Skipping TensorBoard launch (--no-launch specified).")
            logger.info("To launch TensorBoard manually, run:")
            logger.info(f"  tensorboard --logdir {run_root} --port {args.port}")
            return
        launch_tensorboard(run_root, args.port)
    else:
        logger.error("No log directory specified or found")


if __name__ == "__main__":
    main()

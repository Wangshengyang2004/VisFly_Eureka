#!/usr/bin/env python3
"""
Comprehensive test script for all algorithms (BPTT, PPO, SHAC) across environments
"""

import subprocess
import sys
from pathlib import Path
import json
import time

def run_command(cmd, description):
    """Run a command and return success status"""
    print(f"\n{'='*60}")
    print(f"Testing: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        # Check for success indicators
        if "EVALUATION SUMMARY" in result.stdout or "Training completed" in result.stdout:
            print(f"✅ SUCCESS: {description}")

            # Extract key metrics if available
            for line in result.stdout.split('\n'):
                if "Success Rate" in line or "Average Episode Reward" in line or "saved" in line:
                    print(f"   {line.strip()}")

            return True
        else:
            print(f"❌ FAILED: {description}")
            if "Error" in result.stderr or "Error" in result.stdout:
                # Print error lines
                for line in (result.stderr + result.stdout).split('\n'):
                    if "Error" in line or "error" in line.lower():
                        print(f"   Error: {line.strip()}")
            return False

    except subprocess.TimeoutExpired:
        print(f"⏱️ TIMEOUT: {description} (exceeded 120s)")
        return False
    except Exception as e:
        print(f"❌ EXCEPTION: {description} - {str(e)}")
        return False

def test_bptt_environments():
    """Test BPTT with available models"""
    print("\n" + "="*80)
    print("TESTING BPTT ALGORITHM")
    print("="*80)

    test_cases = [
        ("hover", "BPTT_hover_bptt_1.zip"),
        ("navigation", "BPTT_navigation_bptt_2.zip"),
        ("landing", "BPTT_landing_bptt_2.zip"),
        ("object_tracking", "BPTT_object_tracking_bptt_1.zip"),
        ("tracking", "BPTT_tracking_bptt_1.zip"),
        ("flip", "BPTT_flip_bptt_1.zip"),
    ]

    results = []
    for env, weight in test_cases:
        weight_path = Path(f"results/training/{env}/{weight}")
        if not weight_path.exists():
            print(f"⚠️ Skipping {env}: Model {weight} not found")
            continue

        cmd = [
            "python", "run.py",
            "--env", env,
            "--algorithm", "bptt",
            "--train", "0",
            "--weight", weight,
            "--num_eval_episodes", "1",
            "--no_video",  # Skip video for speed
            "--no_plots"   # Skip plots for speed
        ]

        success = run_command(cmd, f"BPTT evaluation on {env}")
        results.append((env, "BPTT", success))

    return results

def test_ppo():
    """Test PPO training and evaluation"""
    print("\n" + "="*80)
    print("TESTING PPO ALGORITHM")
    print("="*80)

    results = []

    # Test PPO training
    print("\n--- Testing PPO Training ---")
    cmd = [
        "python", "run.py",
        "--env", "hover",
        "--algorithm", "ppo",
        "--train", "1",
        "--learning_steps", "500",
        "--comment", "ppo_test_all"
    ]

    success = run_command(cmd, "PPO training on hover")
    results.append(("hover", "PPO-train", success))

    # Test PPO evaluation if training succeeded
    if success:
        print("\n--- Testing PPO Evaluation ---")
        # Find the saved model
        ppo_model = Path("saved/ppo_ppo_test_all_1.zip")
        if ppo_model.exists():
            cmd = [
                "python", "run.py",
                "--env", "hover",
                "--algorithm", "ppo",
                "--train", "0",
                "--weight", str(ppo_model),
                "--num_eval_episodes", "2",
                "--plot_trajectories"  # Test with trajectory plotting
            ]

            success = run_command(cmd, "PPO evaluation with trajectories")
            results.append(("hover", "PPO-eval", success))

    return results

def test_shac():
    """Test SHAC evaluation"""
    print("\n" + "="*80)
    print("TESTING SHAC ALGORITHM")
    print("="*80)

    results = []

    # Test SHAC with existing model
    shac_model = "SHAC_hover_shac_2.zip"
    weight_path = Path(f"results/training/hover/{shac_model}")

    if weight_path.exists():
        cmd = [
            "python", "run.py",
            "--env", "hover",
            "--algorithm", "shac",
            "--train", "0",
            "--weight", shac_model,
            "--num_eval_episodes", "2",
            "--record_video",  # Test with video recording
            "--plot_trajectories"
        ]

        success = run_command(cmd, "SHAC evaluation with video and trajectories")
        results.append(("hover", "SHAC-eval", success))
    else:
        print(f"⚠️ SHAC model {shac_model} not found")

    return results

def main():
    """Run all tests and summarize results"""
    print("\n" + "🚀 "*20)
    print("COMPREHENSIVE ALGORITHM TESTING")
    print("🚀 "*20)

    all_results = []

    # Test all algorithms
    all_results.extend(test_bptt_environments())
    all_results.extend(test_ppo())
    all_results.extend(test_shac())

    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    success_count = sum(1 for _, _, success in all_results if success)
    total_count = len(all_results)

    print(f"\nResults: {success_count}/{total_count} tests passed")
    print("\nDetailed Results:")
    for env, algo, success in all_results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status} - {algo:12s} on {env}")

    # Check if visualization files were created
    print("\n" + "="*80)
    print("VISUALIZATION FILES CHECK")
    print("="*80)

    plots_dir = Path("results/testing/hover/plots")
    videos_dir = Path("results/testing/hover/videos")

    if plots_dir.exists():
        plot_files = list(plots_dir.glob("*.png"))
        print(f"📊 Trajectory plots created: {len(plot_files)} files")

    if videos_dir.exists():
        video_files = list(videos_dir.glob("*.mp4"))
        print(f"🎬 Videos created: {len(video_files)} files")

    return success_count == total_count

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
#!/usr/bin/env python3
"""
Plot Eureka efficacy: success rate and episode length over iterations.
Usage: python scripts/plot_efficacy.py [results_dir]
Default results_dir: results/2026-01-31_20-47-50
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_iteration_metrics(result_root: Path):
    """Load per-iteration metrics from result.json files."""
    train_dir = result_root / "train"
    iters = sorted([int(d.name.replace("iter", "")) for d in train_dir.iterdir() if d.is_dir() and d.name.startswith("iter")])
    data = []
    for i in iters:
        iter_dir = train_dir / f"iter{i}"
        success_count = 0
        success_rates = []
        episode_lengths = []
        final_rewards = []
        for res_file in sorted(iter_dir.glob("sample*/result.json")):
            with open(res_file) as f:
                d = json.load(f)
            sr = d.get("success_rate", -1)
            el = d.get("episode_length", 256)
            fr = d.get("final_reward", 0)
            if sr == 1.0:
                success_count += 1
                episode_lengths.append(el)
                final_rewards.append(fr)
            success_rates.append(sr)
        n_samples = len(success_rates)
        success_rate_pct = (success_count / n_samples * 100) if n_samples else 0
        mean_ep_len = np.mean(episode_lengths) if episode_lengths else np.nan
        best_ep_len = np.min(episode_lengths) if episode_lengths else np.nan
        mean_reward = np.mean(final_rewards) if final_rewards else np.nan
        data.append({
            "iter": i,
            "success_count": success_count,
            "n_samples": n_samples,
            "success_rate_pct": success_rate_pct,
            "mean_episode_length": mean_ep_len,
            "best_episode_length": best_ep_len,
            "mean_final_reward": mean_reward,
        })
    return data


def plot_efficacy(data, out_path: Path):
    """Two-panel figure: success rate and episode length over iterations."""
    iters = [d["iter"] for d in data]
    success_rate = [d["success_rate_pct"] for d in data]
    mean_ep_len = [d["mean_episode_length"] if not np.isnan(d["mean_episode_length"]) else None for d in data]
    best_ep_len = [d["best_episode_length"] if not np.isnan(d["best_episode_length"]) else None for d in data]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
    fig.suptitle("Eureka efficacy (hover task, 15 iterations × 5 samples)", fontsize=11)

    # Panel 1: Success rate (%)
    ax1.bar(iters, success_rate, color="steelblue", edgecolor="navy", alpha=0.85)
    ax1.axhline(y=100, color="gray", linestyle="--", alpha=0.5)
    ax1.set_ylabel("Success rate (%)")
    ax1.set_ylim(0, 105)
    ax1.set_xlabel("")
    ax1.grid(axis="y", alpha=0.3)

    # Panel 2: Episode length (lower = better)
    ax2.plot(iters, [x if x is not None else np.nan for x in mean_ep_len], "o-", label="Mean (success only)", color="green", markersize=6)
    ax2.plot(iters, [x if x is not None else np.nan for x in best_ep_len], "s-", label="Best (min steps)", color="darkgreen", markersize=5)
    ax2.set_ylabel("Episode length (steps)")
    ax2.set_xlabel("Iteration")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.set_ylim(0, 280)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def main():
    root = Path(__file__).resolve().parents[1]
    if len(sys.argv) > 1:
        result_dir = Path(sys.argv[1])
    else:
        result_dir = root / "results" / "2026-01-31_20-47-50"
    if not result_dir.exists():
        print(f"Result dir not found: {result_dir}")
        sys.exit(1)
    data = load_iteration_metrics(result_dir)
    out_path = result_dir / "efficacy_plots.png"
    plot_efficacy(data, out_path)


if __name__ == "__main__":
    main()

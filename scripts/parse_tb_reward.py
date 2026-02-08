#!/usr/bin/env python3
"""Parse TensorBoard events (tensorboard lib only, no TF) and analyze reward trend."""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # suppress TF stdout if tensorboard pulls it in

import sys
from pathlib import Path

# project root
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    logdir = ROOT / "results/2026-02-05_09-37-49/train/iter0/sample1/tensorboard/BPTT_std_1"
    if not logdir.is_dir():
        print(f"Not a directory: {logdir}")
        sys.exit(1)

    event_acc = EventAccumulator(str(logdir))
    event_acc.Reload()

    tags = event_acc.Tags()
    scalars = tags.get("scalars", [])
    print("Scalar tags:", scalars)

    # Prefer reward-related tags; always include ep_rew_mean and ep_success_rate
    reward_tags = [t for t in scalars if "reward" in t.lower() or "success" in t.lower() or "ep_rew" in t or "ep_success" in t]
    if not reward_tags:
        reward_tags = scalars[:5]

    for tag in reward_tags:
        events = event_acc.Scalars(tag)
        if not events:
            continue
        steps = [e.step for e in events]
        vals = [e.value for e in events]
        n = len(vals)
        print(f"\n--- {tag} (n={n}) ---")
        print(f"  step range: {min(steps)} .. {max(steps)}")
        print(f"  value: min={min(vals):.4f} max={max(vals):.4f} mean={sum(vals)/n:.4f}")
        # last 20
        last = vals[-20:] if n >= 20 else vals
        print(f"  last {len(last)}: {[round(v, 4) for v in last]}")

    # Reward trend: rollout/ep_rew_mean
    tag = "rollout/ep_rew_mean"
    if tag in scalars:
        events = event_acc.Scalars(tag)
        if len(events) >= 2:
            vals = [e.value for e in events]
            steps = [e.step for e in events]
            mid = len(vals) // 2
            first_avg = sum(vals[:mid]) / mid
            second_avg = sum(vals[mid:]) / (len(vals) - mid)
            print(f"\n--- Reward trend ({tag}) ---")
            print(f"  first half avg = {first_avg:.4f}, second half avg = {second_avg:.4f}")
            print(f"  step range: {steps[0]} .. {steps[-1]}")
            # coarse buckets
            bucket = max(1, len(vals) // 5)
            for i in range(0, min(5 * bucket, len(vals)), bucket):
                seg = vals[i : i + bucket]
                if seg:
                    print(f"  segment steps ~{steps[i]}-{steps[min(i+bucket,len(steps)-1)]}: mean={sum(seg)/len(seg):.4f}")


if __name__ == "__main__":
    main()

"""
Minimal MATMED smoke test.

Validates end-to-end startup and one training step:
  - Model/runner initialization
  - One episode with one step
  - Metrics CSV write
"""

import argparse
import sys

import torch

from train_matmed import MATMEDRunner
from reward import RewardConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MATMED smoke test")
    parser.add_argument("--output", default="smoke_metrics.csv", help="Metrics CSV path")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    runner = MATMEDRunner(
        reward_config=RewardConfig(alpha=1.0, beta=0.5, gamma=0.5, delta=1.0),
        num_pretrain_epochs=0,
        lr_policy=1e-4,
        gamma=0.99,
        entropy_coeff=0.01,
        seed=args.seed,
        use_chemberta=False,
        device=torch.device("cpu"),
    )

    runner.train(
        num_episodes=1,
        steps_per_episode=1,
        save_csv=args.output,
    )

    if not runner.all_metrics:
        raise RuntimeError("Smoke test failed: no metrics were recorded.")

    ep_metrics = runner.all_metrics[-1]
    print("Smoke test passed.")
    print(f"avg_reward={ep_metrics['avg_reward']:.4f}")
    print(f"best_reward={runner.best_reward:.4f}")
    print(f"best_smiles={runner.best_smiles or '<none>'}")
    print(f"metrics_csv={args.output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Smoke test failed: {exc}", file=sys.stderr)
        raise

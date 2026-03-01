"""
phase3_ablation.py — Phase 3 Ablation Study
=============================================
Compares three MATMED configurations to quantify the Vision-Agent's contribution.
Each variant is now controlled directly via MATMEDRunner parameters (no monkey-patching):

  A. Full MATMED    — use_vision=True,  uncertainty_lambda=0.1
  B. No Vision      — use_vision=False, uncertainty_lambda=0.0
  C. No Uncertainty — use_vision=True,  uncertainty_lambda=0.0
"""

from reward import RewardConfig
from train_matmed import MATMEDRunner


EXPERIMENTS = [
    {
        "name":               "Full_MATMED",
        "use_vision":         True,
        "uncertainty_lambda": 0.1,
    },
    {
        "name":               "No_Vision",
        "use_vision":         False,
        "uncertainty_lambda": 0.0,
    },
    {
        "name":               "No_Uncertainty",
        "use_vision":         True,
        "uncertainty_lambda": 0.0,
    },
]


def run_experiment(
    name: str,
    use_vision: bool,
    uncertainty_lambda: float,
    pretrain_epochs: int = 20,
    num_episodes: int = 30,
    steps_per_episode: int = 8,
    seed: int = 42,
) -> str:
    csv_file = f"phase3_ablation_{name.lower()}.csv"

    print(f"\n{'='*60}")
    print(f"  Ablation: {name}")
    print(f"  use_vision={use_vision}  uncertainty_lambda={uncertainty_lambda}")
    print(f"{'='*60}\n")

    reward_cfg = RewardConfig(alpha=1.0, beta=1.0, gamma=1.0, delta=1.0)

    runner = MATMEDRunner(
        reward_config=reward_cfg,
        num_pretrain_epochs=pretrain_epochs,
        seed=seed,
        use_chemberta=False,
        use_vision=use_vision,
        uncertainty_lambda=uncertainty_lambda,
    )

    runner.train(
        num_episodes=num_episodes,
        steps_per_episode=steps_per_episode,
        save_csv=csv_file,
    )

    print(f"\n  Finished {name} → {csv_file}")
    return csv_file


if __name__ == '__main__':
    results = {}
    for exp in EXPERIMENTS:
        results[exp["name"]] = run_experiment(**exp)

    print("\n\nAll ablation experiments complete!")
    for name, path in results.items():
        print(f"  {name:20s} → {path}")

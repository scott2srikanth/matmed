"""
phase3_ablation.py — Phase 3 Ablation Study
=============================================
Compares three MATMED configurations to quantify the Vision-Agent's contribution:

  A. Full MATMED       — Vision + MC Dropout uncertainty   (λ=0.1)
  B. No Vision         — MLP only, no vision cross-attention
  C. No Uncertainty    — Vision present, but λ=0 (no uncertainty penalty)

Each experiment runs with identical seed, pretrain epochs, and RL episodes.
Results are saved to separate CSV files for plotting.
"""

import os
import copy
import torch
from reward import RewardConfig
from train_matmed import MATMEDRunner
from utils import set_seed


def run_phase3_ablation(
    experiment_name: str,
    use_vision: bool,
    uncertainty_lambda: float,
    pretrain_epochs: int = 20,
    num_episodes: int = 30,
    steps_per_episode: int = 8,
    seed: int = 42,
) -> str:
    """
    Run one ablation experiment variant.

    Args:
        experiment_name:    Label used for CSV filename and console output.
        use_vision:         If False, R-Agent uses MLP only (no VTT cross-attention).
        uncertainty_lambda: λ in R = μ - λσ. Set 0.0 to disable uncertainty penalty.
        pretrain_epochs:    G-Agent pretraining epochs.
        num_episodes:       Number of RL training episodes.
        steps_per_episode:  Steps per episode.
        seed:               Random seed for reproducibility.

    Returns:
        Path to the saved CSV file.
    """
    print(f"\n{'='*60}")
    print(f"  Ablation: {experiment_name}")
    print(f"  use_vision={use_vision}  lambda={uncertainty_lambda}")
    print(f"{'='*60}\n")

    csv_file = f"phase3_ablation_{experiment_name.lower().replace(' ', '_')}.csv"

    reward_cfg = RewardConfig(alpha=1.0, beta=1.0, gamma=1.0, delta=1.0)

    # Monkey-patch the ReactionAgent constructor for this ablation experiment
    # so that we can change use_vision / uncertainty_lambda without touching the runner
    import reaction_agent as ra_module
    OriginalReactionAgent = ra_module.ReactionAgent

    class PatchedReactionAgent(OriginalReactionAgent):
        def __init__(self, *args, **kwargs):
            kwargs['use_vision'] = use_vision
            kwargs['uncertainty_lambda'] = uncertainty_lambda
            super().__init__(*args, **kwargs)

    ra_module.ReactionAgent = PatchedReactionAgent

    try:
        runner = MATMEDRunner(
            reward_config=reward_cfg,
            num_pretrain_epochs=pretrain_epochs,
            seed=seed,
            use_chemberta=False,
        )
        runner.train(
            num_episodes=num_episodes,
            steps_per_episode=steps_per_episode,
            save_csv=csv_file,
        )
    finally:
        # Always restore the original class
        ra_module.ReactionAgent = OriginalReactionAgent

    print(f"\n  ✅ {experiment_name} complete → {csv_file}")
    return csv_file


if __name__ == '__main__':
    results = {}

    # A. Full MATMED: Vision + Uncertainty
    results['Full_MATMED'] = run_phase3_ablation(
        experiment_name='Full_MATMED',
        use_vision=True,
        uncertainty_lambda=0.1,
    )

    # B. No Vision: MLP only, no VTT cross-attention, no uncertainty
    results['No_Vision'] = run_phase3_ablation(
        experiment_name='No_Vision',
        use_vision=False,
        uncertainty_lambda=0.0,
    )

    # C. No Uncertainty: Vision is active, but λ=0 disables the penalty term
    results['No_Uncertainty'] = run_phase3_ablation(
        experiment_name='No_Uncertainty',
        use_vision=True,
        uncertainty_lambda=0.0,
    )

    print("\n\nAll ablation experiments complete!")
    print("CSV files:")
    for name, path in results.items():
        print(f"  {name:20s} → {path}")

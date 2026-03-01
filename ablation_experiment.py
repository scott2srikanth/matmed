"""
ablation_experiment.py — MATMED Ablation Study Runner
======================================================
This script runs the MATMED training loop multiple times with different
agents "turned off" (their weights in the reward function set to 0.0)
to experimentally validate the contribution of each agent to the final
designed molecules.

It produces separate CSV metric files for:
  1. Full MATMED
  2. MATMED (No S-Agent / No Toxicity Critic)
  3. MATMED (No R-Agent / No Yield Critic)
"""

import os
import torch
from train_matmed import MATMEDRunner
from reward import RewardConfig
from utils import set_seed, get_zinc_sample

def run_ablation(experiment_name: str, reward_cfg: RewardConfig, epochs: int = 5, steps: int = 30):
    print(f"\n========================================================")
    print(f"Starting Ablation: {experiment_name}")
    print(f"========================================================\n")
    
    csv_file = f"metrics_{experiment_name.lower().replace(' ', '_')}.csv"
    
    runner = MATMEDRunner(
        reward_config=reward_cfg,
        num_pretrain_epochs=epochs,
        seed=42
    )

    # Train RL Loop
    runner.train(
        num_episodes=steps,
        steps_per_episode=8,
        save_csv=csv_file
    )
    print(f"Finished {experiment_name}. Metrics saved to {csv_file}")


if __name__ == '__main__':
    # 1. Background downloads 2000 ZINC samples to pretrain internally
    zinc_subset = get_zinc_sample(2000)

    # Experiment A: Full MATMED
    reward_full = RewardConfig(alpha=1.0, beta=1.0, gamma=1.0, delta=1.0)
    run_ablation("Full_MATMED", reward_full, epochs=20, steps=30)

    # Experiment B: No Safety Agent (delta=0.0)
    # The P-Agent still sees the toxicity embedding, but the reward doesn't penalize it!
    reward_no_safety = RewardConfig(alpha=1.0, beta=1.0, gamma=1.0, delta=0.0)
    run_ablation("No_SAgent", reward_no_safety, epochs=20, steps=30)

    # Experiment C: No Reaction Agent (beta=0.0)
    # High binding but completely unsynthesizable molecules should emerge.
    reward_no_yield = RewardConfig(alpha=1.0, beta=0.0, gamma=1.0, delta=1.0)
    run_ablation("No_RAgent", reward_no_yield, epochs=20, steps=30)

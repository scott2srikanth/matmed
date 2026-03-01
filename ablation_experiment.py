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
from train_matmed import train_rl_loop, pretrain_generator
from generator_agent import GeneratorAgent
from evaluator_agent import EvaluatorAgent
from safety_agent import SafetyAgent
from reaction_agent import ReactionAgent
from policy_agent import PolicyAgent
from reward import MultiObjectiveReward
from utils import set_seed, get_zinc_sample

def run_ablation(experiment_name: str, reward_fn: MultiObjectiveReward, zinc_sample: list, epochs: int = 5, steps: int = 30):
    print(f"\n========================================================")
    print(f"Starting Ablation: {experiment_name}")
    print(f"========================================================\n")
    
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Re-initialize agents from scratch so they don't share learned weights
    g_agent = GeneratorAgent().to(device)
    e_agent = EvaluatorAgent().to(device)
    s_agent = SafetyAgent().to(device)
    r_agent = ReactionAgent().to(device)
    p_agent = PolicyAgent().to(device)

    # Pretrain G-Agent exactly as in the standard script
    print(f"Pretraining Generator for {experiment_name}...")
    pretrain_generator(g_agent, zinc_sample, num_epochs=epochs, device=device)

    # Train RL Loop
    csv_file = f"metrics_{experiment_name.lower().replace(' ', '_')}.csv"
    train_rl_loop(
        g_agent, e_agent, s_agent, r_agent, p_agent, reward_fn,
        num_episodes=steps, steps_per_episode=8,
        batch_size=4, csv_file=csv_file, device=device
    )
    print(f"Finished {experiment_name}. Metrics saved to {csv_file}")


if __name__ == '__main__':
    # 1. Download/Load 200 ZINC samples for a quick ablation test
    # (Increase to 10000+ for the full paper version)
    zinc_subset = get_zinc_sample(200)

    # Experiment A: Full MATMED
    reward_full = MultiObjectiveReward(alpha=1.0, beta=1.0, gamma=1.0, delta=1.0)
    run_ablation("Full_MATMED", reward_full, zinc_subset, epochs=5, steps=30)

    # Experiment B: No Safety Agent (delta=0.0)
    # The P-Agent still sees the toxicity embedding, but the reward doesn't penalize it!
    reward_no_safety = MultiObjectiveReward(alpha=1.0, beta=1.0, gamma=1.0, delta=0.0)
    run_ablation("No_SAgent", reward_no_safety, zinc_subset, epochs=5, steps=30)

    # Experiment C: No Reaction Agent (beta=0.0)
    # High binding but completely unsynthesizable molecules should emerge.
    reward_no_yield = MultiObjectiveReward(alpha=1.0, beta=0.0, gamma=1.0, delta=1.0)
    run_ablation("No_RAgent", reward_no_yield, zinc_subset, epochs=5, steps=30)

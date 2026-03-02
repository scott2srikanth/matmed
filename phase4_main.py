"""
phase4_main.py — MATMED Phase 4 RL Loop & Ablations
===================================================
1. Loads all 4 pre-trained Phase 4 agents (G, E, S, R).
2. Freezes the critics (E, S, R).
3. Evaluates the multi-objective reward function using Phase 4 normalizations.
4. Explores the generation space via PPO.
5. Runs 500 iterations of ablation experiments (Full vs No-Safety vs No-Reaction vs No-Binding).
"""

import os
import torch
import pandas as pd
import numpy as np

from utils import get_logger, set_seed
from reward import RewardConfig, RewardFunction
from train_matmed import MATMEDRunner
from evaluator_agent import EvaluatorAgent
from safety_agent import SafetyAgent
from reaction_agent import ReactionAgent
from generator_agent import GeneratorAgent
from utils import SMILESTokenizer

logger = get_logger('Phase4_RL')

def load_pretrained_runner(
    ablation_config: dict,
    seed: int = 42,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> MATMEDRunner:
    """Creates a runner but swaps in the pre-trained weights instead of scratch models."""
    
    # Base configuration:
    reward_cfg = RewardConfig(
        alpha=1.0 if not ablation_config.get('no_binding') else 0.0,
        beta=1.0  if not ablation_config.get('no_reaction') else 0.0,
        gamma=1.0 if not ablation_config.get('no_safety') else 0.0,
        delta=1.0 # synthetic complexity penalty handled internally by reaction score right now
    )

    runner = MATMEDRunner(
        reward_config=reward_cfg,
        num_pretrain_epochs=0, # Skipping phase 2 generic pretrain!
        seed=seed,
        use_chemberta=False,
        use_vision=not ablation_config.get('no_vision', False),
        uncertainty_lambda=0.1,
        lr_policy=1e-5,        # Adjusted from 5e-5 to 1e-5
        entropy_coeff=0.1,     # Adjusted from 0.05 to 0.1
        ppo_clip=0.1,          # Adjusted from 0.25 to 0.1
        kl_coef=0.1,           # New KL reg parameter
    )
    
    # Load weights
    try:
        # 1. Generator
        logger.info("Loading Phase 4 G-Agent (ChEMBL LM)...")
        if os.path.exists("generator_pretrained.pt"):
            runner.g_agent.load_state_dict(torch.load("generator_pretrained.pt", map_location=device))
        else:
            logger.warning("generator_pretrained.pt not found! Using random Init.")

        # 2. Binding Evaluator
        if not ablation_config.get('no_binding'):
            logger.info("Loading Phase 4 E-Agent (BindingDB pIC50)...")
            if os.path.exists("binding_regressor.pt"):
                runner.e_agent.load_state_dict(torch.load("binding_regressor.pt", map_location=device))
                runner.e_agent.eval()
                for p in runner.e_agent.parameters(): p.requires_grad = False
            else:
                logger.warning("binding_regressor.pt not found!")

        # 3. Safety Critic
        if not ablation_config.get('no_safety'):
            logger.info("Loading Phase 4 S-Agent (Tox21)...")
            if os.path.exists("toxicity_model.pt"):
                # We only saved the encoder backbone!
                runner.s_agent.encoder.load_state_dict(torch.load("toxicity_model.pt", map_location=device))
                runner.s_agent.eval()
                for p in runner.s_agent.parameters(): p.requires_grad = False
            else:
                logger.warning("toxicity_model.pt not found!")

        # 4. Reaction Feasibility
        if not ablation_config.get('no_reaction'):
            logger.info("Loading Phase 4 R-Agent (USPTO)...")
            if os.path.exists("reaction_model.pt"):
                runner.r_agent.load_state_dict(torch.load("reaction_model.pt", map_location=device))
                runner.r_agent.eval()
                for p in runner.r_agent.parameters(): p.requires_grad = False
            else:
                logger.warning("reaction_model.pt not found!")

    except Exception as e:
         logger.error(f"Failed to load a pretrained weight: {e}")

    return runner


def run_phase4_experiment(
    experiment_name: str,
    ablation_config: dict,
    num_episodes: int = 50, # Reduced from 500 for prototype speed, scale up later
    steps_per_episode: int = 64
):
    print(f"\n{'='*70}\n  Phase 4 RL — {experiment_name}\n{'='*70}")
    
    csv_file = f"phase4_metrics_{experiment_name.lower().replace(' ', '_')}.csv"
    runner = load_pretrained_runner(ablation_config)
    
    runner.train(
        num_episodes=num_episodes,
        steps_per_episode=steps_per_episode,
        save_csv=csv_file
    )
    
    # Save top 50 molecules to CSV
    top_50 = pd.DataFrame({'smiles': list(runner.top_smiles.keys())[:50]})
    top_50.to_csv(f"phase4_top50_{experiment_name.lower().replace(' ', '_')}.csv", index=False)
    print(f"Finished {experiment_name}. Saved {csv_file}")
    return csv_file

if __name__ == '__main__':
    ablations = {
        "Full MATMED":        {},
        "No Safety Agent":    {"no_safety": True},
        "No Reaction Agent":  {"no_reaction": True},
        "No Binding Agent":   {"no_binding": True},
        "No Vision Agent":    {"no_vision": True}
    }
    
    results = {}
    for name, cfg in ablations.items():
        results[name] = run_phase4_experiment(name, cfg, num_episodes=50) # Use 50 for prototype

    print("\nPhase 4 RL & Ablations Complete!")

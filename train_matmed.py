"""
train_matmed.py — MATMED Training Loop (Phase 3)
=================================================
Orchestrates the full MATMED pipeline:

  G-Agent → molecule generation
  E-Agent → binding score + graph embedding
  S-Agent → ADMET (toxicity + druglikeness)
  R-Agent → reaction feasibility (yield) ← now with Vision cross-attention!
  V-Agent → reaction video simulation via VisionTemporalTransformer
  P-Agent → policy decision (accept / modify / regenerate / stop)
  Reward  → composite multi-objective signal
  RL      → REINFORCE with value-function baseline

New in Phase 3:
  - Each molecule now triggers simulate_reaction_video() → (50, 16) tensor
  - Vision tensor is passed to R-Agent for cross-attended yield prediction
  - Per-episode logging: vision_variance and yield_sa_corr

Usage::

    python train_matmed.py --num_episodes 50 --steps_per_episode 10
"""

import os
import csv
import argparse
import random
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn as nn

from utils import (
    set_seed, get_logger, SMILESTokenizer, is_valid_smiles,
    diversity_score, get_zinc_sample, save_metrics_csv,
)
from reward import RewardFunction, RewardConfig
from generator_agent import GeneratorAgent, pretrain_generator
from evaluator_agent import EvaluatorAgent
from safety_agent import SafetyAgent
from reaction_agent import ReactionAgent, compute_reaction_features
from vision_agent import simulate_reaction_video
from policy_agent import PolicyAgent, ACTION_NAMES, _discount_returns

logger = get_logger('MATMED')


# ─────────────────────────────────────────────────────────────────────────────
# Transition buffer
# ─────────────────────────────────────────────────────────────────────────────

class Transition:
    """Stores a single RL transition for REINFORCE training."""
    __slots__ = ('log_prob', 'reward', 'value', 'action_probs')

    def __init__(
        self,
        log_prob: float,
        reward: float,
        value: torch.Tensor,
        action_probs: torch.Tensor,
    ) -> None:
        self.log_prob     = log_prob
        self.reward       = reward
        self.value        = value
        self.action_probs = action_probs


# ─────────────────────────────────────────────────────────────────────────────
# MATMED Runner
# ─────────────────────────────────────────────────────────────────────────────

class MATMEDRunner:
    """
    Top-level MATMED training harness.

    Holds all agents, the reward function, and training state.
    Provides episode-level run loops and metric logging.
    """

    def __init__(
        self,
        reward_config: Optional[RewardConfig] = None,
        num_pretrain_epochs: int = 5,
        lr_policy: float = 3e-4,
        gamma: float = 0.99,
        entropy_coeff: float = 0.01,
        seed: int = 42,
        device: Optional[torch.device] = None,
        use_chemberta: bool = False,
        use_vision: bool = True,              # Phase 3: enable VTT cross-attention
        uncertainty_lambda: float = 0.1,      # Phase 3: lambda in R = mu - lambda*sigma
        ppo_clip: float = 1.0,                # Phase 4: gradient clip tuning
    ) -> None:
        set_seed(seed)

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        logger.info(f"MATMED running on {device}")

        # ── Agents ──────────────────────────────────────────────────────────
        self.tokenizer = SMILESTokenizer()

        self.g_agent = GeneratorAgent(tokenizer=self.tokenizer).to(device)
        self.e_agent = EvaluatorAgent().to(device)
        self.s_agent = SafetyAgent(use_chemberta=use_chemberta).to(device)
        self.r_agent = ReactionAgent(
            use_vision=use_vision,
            uncertainty_lambda=uncertainty_lambda,
        ).to(device)

        self.p_agent = PolicyAgent(
            d_g=self.g_agent.d_model,
            d_e=self.e_agent.hidden_dim,
            d_s=self.s_agent.embed_dim,
            d_r=self.r_agent.hidden_dim,
        ).to(device)

        # ── Reward ──────────────────────────────────────────────────────────
        self.reward_fn = RewardFunction(reward_config)

        # ── Optimisers ──────────────────────────────────────────────────────
        self.policy_optim = torch.optim.Adam(self.p_agent.parameters(), lr=lr_policy)

        # ── Hyper-params ────────────────────────────────────────────────────
        self.gamma = gamma
        self.entropy_coeff = entropy_coeff
        self.ppo_clip = ppo_clip

        # ── Metrics ─────────────────────────────────────────────────────────
        self.best_reward    = -float('inf')
        self.all_metrics: List[Dict] = []
        self.top_smiles: Dict[str, float] = {}  # Track top N SMILES

        # ── Pretrain G-Agent ────────────────────────────────────────────────
        logger.info("Pretraining G-Agent on ZINC sample …")
        zinc_smiles = get_zinc_sample()
        pretrain_generator(self.g_agent, zinc_smiles, num_epochs=num_pretrain_epochs)
        logger.info("G-Agent pretraining complete.")

    # ── Episode step ────────────────────────────────────────────────────────

    def _step(self, prev_reward: float = 0.0) -> Tuple[str, float, Dict, Transition]:
        """
        Run one full MATMED step:
          1. Generate SMILES (G-Agent)
          2. Predict binding (E-Agent)
          3. Evaluate ADMET (S-Agent)
          4. Estimate yield (R-Agent)
          5. Compute reward
          6. P-Agent selects action

        Returns:
            smiles:     Generated SMILES string.
            reward:     Scalar reward.
            scores:     Dict of individual scores.
            transition: RL Transition object.
        """
        # 1. Generate ─────────────────────────────────────────────────────
        self.g_agent.eval()
        gen_smiles_list, g_emb = self.g_agent.generate(batch_size=1, temperature=1.0)
        smiles = gen_smiles_list[0]
        g_emb  = g_emb[0].detach()  # (d_g,)

        valid = is_valid_smiles(smiles)

        # 2. Binding (E-Agent) ─────────────────────────────────────────────
        try:
            binding_score, e_emb = self.e_agent.forward(smiles if valid else 'C')
            e_emb = e_emb.detach()
        except Exception:
            binding_score = 0.0
            e_emb = torch.zeros(self.e_agent.hidden_dim, device=self.device)

        # 3. ADMET (S-Agent) ───────────────────────────────────────────────
        try:
            toxicity, admet_score, s_emb = self.s_agent.forward(smiles if valid else 'C')
            s_emb = s_emb.detach()
        except Exception:
            toxicity, admet_score = 0.5, 0.5
            s_emb = torch.zeros(self.s_agent.embed_dim, device=self.device)

        # 4. Yield (R-Agent) + Vision ────────────────────────────────────
        mol_smiles  = smiles if valid else 'C'
        mol_feats   = compute_reaction_features(mol_smiles)

        vision_var  = 0.0
        vis_seq     = None
        if mol_feats is not None:
            vis_seq    = simulate_reaction_video(mol_feats).to(self.device)  # (50, 16)
            vision_var = float(torch.var(vis_seq).item())

        yield_score, r_emb = self.r_agent.forward(mol_smiles, vision_seq=vis_seq)
        r_emb = r_emb.detach()

        # 5. Reward ───────────────────────────────────────────────────────
        if not valid:
            reward = -1.0   # hard penalty for invalid molecule
        else:
            reward = self.reward_fn.compute(binding_score, yield_score, admet_score, toxicity)

        scores = {
            'binding':           binding_score,
            'yield':             yield_score,
            'admet':             admet_score,
            'toxicity':          toxicity,
            'vision_var':        vision_var,
        }

        # Also log raw uncertainty (mu, sigma) for diagnostics
        if mol_feats is not None:
            mu, sigma = self.r_agent.get_uncertainty(mol_smiles, vision_seq=vis_seq)
            scores['yield_mu']    = mu
            scores['yield_sigma'] = sigma

        # 6. Policy ───────────────────────────────────────────────────────
        self.p_agent.train()
        logits, action_probs_t, value_t = self.p_agent.forward(
            g_emb, e_emb, s_emb, r_emb, reward=prev_reward
        )
        dist   = torch.distributions.Categorical(action_probs_t)
        action_t = dist.sample()
        log_prob = dist.log_prob(action_t).item()

        transition = Transition(
            log_prob    = log_prob,
            reward      = reward,
            value       = value_t.squeeze(),
            action_probs= action_probs_t.squeeze().detach(),
        )

        action_name = ACTION_NAMES[action_t.item() % 4]
        logger.debug(
            f"  smiles={smiles[:30]}  valid={valid}  "
            f"bind={binding_score:.3f}  yield={yield_score:.3f}  "
            f"admet={admet_score:.3f}  tox={toxicity:.3f}  "
            f"reward={reward:.3f}  action={action_name}"
        )

        return smiles, reward, scores, transition

    # ── Policy update ───────────────────────────────────────────────────────

    def _update_policy(self, transitions: List[Transition]) -> Dict[str, float]:
        """Run REINFORCE update over one episode's transitions."""
        rewards   = torch.tensor([t.reward    for t in transitions], dtype=torch.float, device=self.device)
        log_probs = torch.tensor([t.log_prob  for t in transitions], dtype=torch.float, device=self.device)
        values    = torch.stack([t.value      for t in transitions])
        probs_all = torch.stack([t.action_probs for t in transitions])

        loss_dict = self.p_agent.compute_policy_loss(
            log_probs, rewards, values,
            gamma=self.gamma,
            entropy_coeff=self.entropy_coeff,
            action_probs_all=probs_all,
        )

        self.policy_optim.zero_grad()
        loss_dict['total_loss'].backward()
        nn.utils.clip_grad_norm_(self.p_agent.parameters(), max_norm=self.ppo_clip)
        self.policy_optim.step()

        return {k: float(v.item()) for k, v in loss_dict.items()}

    # ── Episode runner ──────────────────────────────────────────────────────

    def run_episode(
        self,
        episode_idx: int,
        max_steps: int = 10,
    ) -> Dict[str, float]:
        """
        Run a single RL episode.

        Args:
            episode_idx: Episode number (for logging).
            max_steps:   Maximum steps before forcing termination.

        Returns:
            Aggregated episode metrics.
        """
        transitions: List[Transition] = []
        episode_smiles: List[str]     = []
        episode_rewards: List[float]  = []
        episode_yields:  List[float]  = []
        episode_sa:      List[float]  = []
        episode_vis_var: List[float]  = []
        episode_sigma:   List[float]  = []   # yield uncertainty

        prev_reward = 0.0
        for step in range(max_steps):
            smiles, reward, scores, transition = self._step(prev_reward=prev_reward)
            transitions.append(transition)
            episode_smiles.append(smiles)
            episode_rewards.append(reward)
            episode_yields.append(scores.get('yield', 0.0))
            episode_vis_var.append(scores.get('vision_var', 0.0))
            episode_sigma.append(scores.get('yield_sigma', 0.0))
            prev_reward = reward

            # Accumulate per-molecule SA score for correlation
            from reaction_agent import compute_reaction_features
            mol_feats = compute_reaction_features(smiles if is_valid_smiles(smiles) else 'C')
            sa_score = float(mol_feats[0]) if mol_feats is not None else 0.0
            episode_sa.append(sa_score)

            # Track best
            if is_valid_smiles(smiles):
                if reward > self.best_reward:
                    self.best_reward = reward
                    self.best_smiles = smiles
                    logger.info(f"  ★ New best reward={reward:.4f}  SMILES={smiles}")
                
                # Maintain top 50 unique SMILES by reward
                if smiles not in self.top_smiles or reward > self.top_smiles[smiles]:
                    self.top_smiles[smiles] = reward
                if len(self.top_smiles) > 50:
                    # Remove the lowest reward item
                    lowest_smi = min(self.top_smiles, key=self.top_smiles.get)
                    del self.top_smiles[lowest_smi]

        # Policy update
        loss_info = self._update_policy(transitions)

        # Vision Metrics ─────────────────────────────────────────────────────
        avg_vis_var      = sum(episode_vis_var) / max(1, len(episode_vis_var))
        avg_yield_sigma  = sum(episode_sigma) / max(1, len(episode_sigma))

        # Pearson correlation between predicted yields and SA scores
        import numpy as np
        yield_arr = np.array(episode_yields, dtype=np.float32)
        sa_arr    = np.array(episode_sa, dtype=np.float32)
        if yield_arr.std() > 1e-6 and sa_arr.std() > 1e-6:
            yield_sa_corr = float(np.corrcoef(yield_arr, sa_arr)[0, 1])
        else:
            yield_sa_corr = 0.0

        # Aggregate metrics
        valid_count  = sum(is_valid_smiles(s) for s in episode_smiles)
        pct_valid    = 100.0 * valid_count / max(1, max_steps)
        avg_reward   = sum(episode_rewards) / max(1, len(episode_rewards))
        diversity    = diversity_score(episode_smiles)

        metrics = {
            'episode':            episode_idx,
            'avg_reward':         avg_reward,
            'best_reward':        self.best_reward,
            'pct_valid':          pct_valid,
            'diversity':          diversity,
            'vision_variance':    avg_vis_var,
            'yield_sa_corr':      yield_sa_corr,
            'yield_uncertainty':  avg_yield_sigma,
            **loss_info,
        }

        logger.info(
            f"Episode {episode_idx:3d} | avg_rwd={avg_reward:.3f} | "
            f"best={self.best_reward:.3f} | valid={pct_valid:.0f}% | "
            f"div={diversity:.3f} | loss={loss_info.get('total_loss', 0):.4f} | "
            f"vis_var={avg_vis_var:.3f} | yield_sa_corr={yield_sa_corr:.3f} | "
            f"yield_unc={avg_yield_sigma:.3f}"
        )

        self.all_metrics.append(metrics)
        return metrics

    # ── Training ────────────────────────────────────────────────────────────

    def train(
        self,
        num_episodes: int = 50,
        steps_per_episode: int = 10,
        save_csv: str = 'matmed_metrics.csv',
    ) -> None:
        """
        Full training loop.

        Args:
            num_episodes:       Total RL episodes.
            steps_per_episode:  Steps per episode.
            save_csv:           Path to save per-episode metrics.
        """
        logger.info(f"Starting MATMED training — {num_episodes} episodes × {steps_per_episode} steps")

        for ep in range(1, num_episodes + 1):
            self.run_episode(ep, max_steps=steps_per_episode)

        save_metrics_csv(self.all_metrics, filepath=save_csv)
        logger.info(f"Metrics saved → {save_csv}")
        logger.info(
            f"\n{'='*60}\nTraining complete!\n"
            f"Best reward : {self.best_reward:.4f}\n"
            f"Best SMILES : {self.best_smiles}\n"
            f"{'='*60}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='MATMED Training')
    p.add_argument('--num_episodes',     type=int,   default=20)
    p.add_argument('--steps_per_episode',type=int,   default=8)
    p.add_argument('--num_pretrain',     type=int,   default=5)
    p.add_argument('--lr',               type=float, default=3e-4)
    p.add_argument('--gamma',            type=float, default=0.99)
    p.add_argument('--entropy_coeff',    type=float, default=0.01)
    p.add_argument('--alpha',            type=float, default=1.0,  help='Binding weight')
    p.add_argument('--beta',             type=float, default=0.5,  help='Yield weight')
    p.add_argument('--gamma_reward',     type=float, default=0.5,  help='ADMET weight')
    p.add_argument('--delta',            type=float, default=1.0,  help='Toxicity penalty')
    p.add_argument('--seed',             type=int,   default=42)
    p.add_argument('--use_chemberta',    action='store_true', default=False)
    p.add_argument('--save_csv',         type=str,   default='matmed_metrics.csv')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()

    reward_cfg = RewardConfig(
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma_reward,
        delta=args.delta,
    )

    runner = MATMEDRunner(
        reward_config=reward_cfg,
        num_pretrain_epochs=args.num_pretrain,
        lr_policy=args.lr,
        gamma=args.gamma,
        entropy_coeff=args.entropy_coeff,
        seed=args.seed,
        use_chemberta=args.use_chemberta,
    )

    runner.train(
        num_episodes=args.num_episodes,
        steps_per_episode=args.steps_per_episode,
        save_csv=args.save_csv,
    )

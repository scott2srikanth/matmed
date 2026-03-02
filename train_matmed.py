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

import numpy as np
import torch
import torch.nn as nn

from utils import (
    set_seed, get_logger, SMILESTokenizer, is_valid_smiles,
    diversity_score, get_zinc_sample, save_metrics_csv,
)
from reward import RewardFunction, RewardConfig, RewardAggregator
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
    """Stores a single RL transition for PPO training."""
    __slots__ = (
        'g_emb',
        'e_emb',
        's_emb',
        'r_emb',
        'prev_reward',
        'action',
        'old_log_prob',
        'old_action_probs',
        'reward',
    )

    def __init__(
        self,
        g_emb: torch.Tensor,
        e_emb: torch.Tensor,
        s_emb: torch.Tensor,
        r_emb: torch.Tensor,
        prev_reward: float,
        action: int,
        old_log_prob: float,
        old_action_probs: torch.Tensor,
        reward: float,
    ) -> None:
        self.g_emb = g_emb
        self.e_emb = e_emb
        self.s_emb = s_emb
        self.r_emb = r_emb
        self.prev_reward = prev_reward
        self.action = action
        self.old_log_prob = old_log_prob
        self.old_action_probs = old_action_probs
        self.reward = reward


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
        lr_policy: float = 1e-5,
        gamma: float = 0.99,
        entropy_coeff: float = 0.1,
        seed: int = 42,
        device: Optional[torch.device] = None,
        use_chemberta: bool = False,
        use_vision: bool = True,              # Phase 3: enable VTT cross-attention
        uncertainty_lambda: float = 0.1,      # Phase 3: lambda in R = mu - lambda*sigma
        ppo_clip: float = 0.1,                # stabilized PPO clip range
        kl_coef: float = 0.1,                 # strong KL anchor
        ppo_epochs: int = 4,
    ) -> None:
        set_seed(seed)

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        logger.info(f"MATMED running on {device}")

        # ── Agents ──────────────────────────────────────────────────────────
        self.tokenizer = SMILESTokenizer()

        self.g_agent = GeneratorAgent(tokenizer=self.tokenizer).to(device)
        self.e_agent = EvaluatorAgent(num_layers=4).to(device)
        self.s_agent = SafetyAgent(use_chemberta=use_chemberta, d_model=256).to(device)
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
        self.reward_aggregator = RewardAggregator(
            w_bind=0.5,
            w_safety=1.0,
            w_reaction=1.0,
            w_vision=0.5,
        )

        # ── Optimisers ──────────────────────────────────────────────────────
        self.policy_optim = torch.optim.Adam(self.p_agent.parameters(), lr=lr_policy)

        # ── Hyper-params ────────────────────────────────────────────────────
        self.gamma = gamma
        self.entropy_coeff = entropy_coeff
        self.ppo_clip = ppo_clip
        self.kl_coef = kl_coef
        self.ppo_epochs = ppo_epochs
        self.target_kl = 0.02
        self.vision_ood_coef = 0.1
        self.vision_embed_mean = None
        self.vision_dist_std = None
        if os.path.exists("vision_embed_stats.npz"):
            try:
                stats = np.load("vision_embed_stats.npz")
                self.vision_embed_mean = torch.tensor(stats["mean"], dtype=torch.float, device=self.device)
                self.vision_dist_std = float(stats["dist_std"])
                logger.info("Loaded vision_embed_stats.npz for vision OOD penalty.")
            except Exception as exc:
                logger.warning(f"Could not load vision_embed_stats.npz: {exc}")

        # ── Metrics ─────────────────────────────────────────────────────────
        self.best_reward    = -float('inf')
        self.best_smiles    = ""
        self.all_metrics: List[Dict] = []
        self.top_smiles: Dict[str, float] = {}  # Track top N SMILES
        self.global_step = 0

        # ── Pretrain G-Agent ────────────────────────────────────────────────
        logger.info("Pretraining G-Agent on ZINC sample …")
        zinc_smiles = get_zinc_sample()
        pretrain_generator(self.g_agent, zinc_smiles, num_epochs=num_pretrain_epochs)
        logger.info("G-Agent pretraining complete.")

    # ── Episode step ────────────────────────────────────────────────────────

    def _step(
        self,
        prev_reward: float = 0.0,
        requested_action: int = 2,
    ) -> Tuple[str, float, Dict, Transition, int]:
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
        if requested_action == 0:          # ACCEPT -> exploit nearby high-probability region
            temperature, top_k = 0.7, 16
        elif requested_action == 1:        # MODIFY -> moderate exploration
            temperature, top_k = 0.9, 32
        else:                              # REGENERATE / default -> broader exploration
            temperature, top_k = 1.2, 0

        gen_smiles_list, g_emb = self.g_agent.generate(
            batch_size=1,
            temperature=temperature,
            top_k=top_k,
        )
        smiles = gen_smiles_list[0]
        g_emb  = g_emb[0].detach()  # (d_g,)

        valid = is_valid_smiles(smiles)

        if not valid:
            reward = -2.0   # hard penalty for invalid molecule
            binding_score, yield_score, admet_score, toxicity = 0.0, 0.0, 0.0, 0.0
            vision_var, mu, sigma = 0.0, 0.0, 0.0
            e_emb = torch.zeros(self.e_agent.hidden_dim, device=self.device)
            s_emb = torch.zeros(self.s_agent.embed_dim, device=self.device)
            r_emb = torch.zeros(self.r_agent.hidden_dim, device=self.device)
            mol_feats = None
            vis_seq = None
        else:
            # 2. Binding (E-Agent) ─────────────────────────────────────────────
            try:
                binding_score, e_emb = self.e_agent.forward(smiles)
                e_emb = e_emb.detach()
            except Exception:
                binding_score = 0.0
                e_emb = torch.zeros(self.e_agent.hidden_dim, device=self.device)

            # 3. ADMET (S-Agent) ───────────────────────────────────────────────
            try:
                toxicity, admet_score, s_emb = self.s_agent.forward(smiles)
                s_emb = s_emb.detach()
            except Exception:
                toxicity, admet_score = 0.5, 0.5
                s_emb = torch.zeros(self.s_agent.embed_dim, device=self.device)

            # 4. Yield (R-Agent) + Vision ────────────────────────────────────
            mol_feats = compute_reaction_features(smiles)
            vision_var = 0.0
            vis_seq = None
            if mol_feats is not None:
                vis_seq = simulate_reaction_video(mol_feats).to(self.device)  # (50, 28)
                vision_var = float(torch.var(vis_seq).item())

            yield_score, r_emb = self.r_agent.forward(smiles, vision_seq=vis_seq)
            r_emb = r_emb.detach()

            # 5. Reward ───────────────────────────────────────────────────────
            # Running-normalized multi-objective aggregation to balance critic scales.
            # Safety component combines drug-likeness and toxicity into one term.
            total_reward_t, reward_stats = self.reward_aggregator.aggregate(
                r_bind_raw=torch.tensor([binding_score], dtype=torch.float, device=self.device),
                r_safety_raw=torch.tensor([admet_score - toxicity], dtype=torch.float, device=self.device),
                r_reaction_raw=torch.tensor([yield_score], dtype=torch.float, device=self.device),
                r_vision_raw=torch.tensor([vision_var], dtype=torch.float, device=self.device),
                return_details=True,
            )
            reward = float(total_reward_t.squeeze().item())

            # Vision OOD penalty using training embedding distribution, if available.
            if (
                self.vision_embed_mean is not None
                and self.vision_dist_std is not None
                and vis_seq is not None
                and self.r_agent.vision_transformer is not None
            ):
                with torch.no_grad():
                    vis_emb = self.r_agent.vision_transformer(vis_seq.unsqueeze(0))
                    dist = torch.norm(vis_emb.squeeze(0) - self.vision_embed_mean, p=2)
                    ood_pen = torch.relu(dist - 2.0 * self.vision_dist_std)
                    reward -= self.vision_ood_coef * float(ood_pen.item())
                    reward_stats['vision_ood_pen'] = ood_pen.detach().view(1)

            # Live critic variance monitoring.
            self.global_step += 1
            if self.global_step % 100 == 0:
                rb = reward_stats['bind']
                rs = reward_stats['safety']
                rr = reward_stats['reaction']
                rv = reward_stats['vision']
                cb = reward_stats['c_bind']
                cs = reward_stats['c_safety']
                cr = reward_stats['c_reaction']
                ood_pen = reward_stats['ood_pen']
                vision_ood_pen = reward_stats.get('vision_ood_pen', torch.zeros_like(ood_pen))
                temps = self.reward_aggregator.calibrator.temp
                logger.info(
                    "Reward stats | bind mean/std: %.4f/%.4f | safety mean/std: %.4f/%.4f | "
                    "reaction mean/std: %.4f/%.4f | vision mean/std: %.4f/%.4f | "
                    "conf(b/s/r): %.3f/%.3f/%.3f | ood(raw/vis): %.4f/%.4f | "
                    "temp(b/s/r/v): %.3f/%.3f/%.3f/%.3f",
                    rb.mean().item(), rb.std(unbiased=False).item(),
                    rs.mean().item(), rs.std(unbiased=False).item(),
                    rr.mean().item(), rr.std(unbiased=False).item(),
                    rv.mean().item(), rv.std(unbiased=False).item(),
                    cb.mean().item(), cs.mean().item(), cr.mean().item(),
                    ood_pen.mean().item(),
                    vision_ood_pen.mean().item(),
                    temps['bind'], temps['safety'], temps['reaction'], temps['vision'],
                )

        scores = {
            'binding':           binding_score,
            'yield':             yield_score,
            'admet':             admet_score,
            'toxicity':          toxicity,
            'vision_var':        vision_var,
        }

        # Also log raw uncertainty (mu, sigma) for diagnostics
        if valid and mol_feats is not None:
            mu, sigma = self.r_agent.get_uncertainty(smiles, vision_seq=vis_seq)
            scores['yield_mu']    = mu
            scores['yield_sigma'] = sigma

        # 6. Policy ───────────────────────────────────────────────────────
        self.p_agent.train()
        logits, action_probs_t, value_t = self.p_agent.forward(
            g_emb, e_emb, s_emb, r_emb, reward=prev_reward
        )
        dist   = torch.distributions.Categorical(action_probs_t)
        action_t = dist.sample()
        log_prob = dist.log_prob(action_t)
        action_idx = int(action_t.item())

        transition = Transition(
            g_emb=g_emb.detach(),
            e_emb=e_emb.detach(),
            s_emb=s_emb.detach(),
            r_emb=r_emb.detach(),
            prev_reward=prev_reward,
            action=action_idx,
            old_log_prob=float(log_prob.detach().item()),
            old_action_probs=action_probs_t.squeeze().detach(),
            reward=reward,
        )

        action_name = ACTION_NAMES[action_idx % 4]
        logger.debug(
            f"  smiles={smiles[:30]}  valid={valid}  "
            f"bind={binding_score:.3f}  yield={yield_score:.3f}  "
            f"admet={admet_score:.3f}  tox={toxicity:.3f}  "
            f"reward={reward:.3f}  action={action_name}"
        )
        scores['action'] = action_idx

        return smiles, reward, scores, transition, action_idx

    # ── Policy update ───────────────────────────────────────────────────────

    def _update_policy(self, transitions: List[Transition]) -> Dict[str, float]:
        """Run PPO update over one episode's transitions."""
        if len(transitions) == 0:
            return {
                'policy_loss': 0.0,
                'value_loss': 0.0,
                'entropy': 0.0,
                'total_loss': 0.0,
                'kl_coef': float(self.kl_coef),
                'kl': 0.0,
                'approx_kl': 0.0,
                'policy_entropy': 0.0,
            }

        rewards = torch.tensor([t.reward for t in transitions], dtype=torch.float, device=self.device)
        old_log_probs = torch.tensor([t.old_log_prob for t in transitions], dtype=torch.float, device=self.device)
        old_action_probs = torch.stack([t.old_action_probs for t in transitions], dim=0).to(self.device)
        old_action_probs = torch.nan_to_num(old_action_probs, nan=0.25, posinf=0.25, neginf=0.25)
        old_action_probs = old_action_probs.clamp(min=1e-8)
        old_action_probs = old_action_probs / old_action_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        actions = torch.tensor([t.action for t in transitions], dtype=torch.long, device=self.device)
        prev_rewards = torch.tensor([t.prev_reward for t in transitions], dtype=torch.float, device=self.device)
        returns = _discount_returns(rewards, gamma=self.gamma)

        g_batch = torch.stack([t.g_emb for t in transitions], dim=0).to(self.device)
        e_batch = torch.stack([t.e_emb for t in transitions], dim=0).to(self.device)
        s_batch = torch.stack([t.s_emb for t in transitions], dim=0).to(self.device)
        r_batch = torch.stack([t.r_emb for t in transitions], dim=0).to(self.device)

        approx_kl = 0.0
        true_kl = 0.0
        final_policy_loss = torch.tensor(0.0, device=self.device)
        final_value_loss = torch.tensor(0.0, device=self.device)
        final_entropy = torch.tensor(0.0, device=self.device)
        final_total_loss = torch.tensor(0.0, device=self.device)
        
        for _ in range(self.ppo_epochs):
            _, action_probs, values = self.p_agent.forward(
                g_batch,
                e_batch,
                s_batch,
                r_batch,
                reward=prev_rewards,
            )
            action_probs = torch.nan_to_num(action_probs, nan=0.25, posinf=0.25, neginf=0.25)
            action_probs = action_probs.clamp(min=1e-8)
            action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            dist = torch.distributions.Categorical(action_probs)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            advantages = returns - values.squeeze(-1).detach()
            adv_mean = advantages.mean()
            if advantages.numel() > 1:
                adv_std = advantages.std(unbiased=False).clamp(min=1e-8)
                advantages = (advantages - adv_mean) / adv_std
            else:
                # Single-step rollout: avoid undefined std normalization.
                advantages = advantages - adv_mean

            ratios = torch.exp(new_log_probs - old_log_probs)
            surrogate1 = ratios * advantages
            surrogate2 = torch.clamp(ratios, 1 - self.ppo_clip, 1 + self.ppo_clip) * advantages
            policy_loss = -torch.min(surrogate1, surrogate2).mean()
            value_loss = torch.nn.functional.mse_loss(values.squeeze(-1), returns)
            approx_kl_t = 0.5 * (new_log_probs - old_log_probs).pow(2).mean()
            kl_t = (
                old_action_probs
                * (torch.log(old_action_probs + 1e-8) - torch.log(action_probs + 1e-8))
            ).sum(dim=-1).mean()
            total_loss = (
                policy_loss
                + 0.5 * value_loss
                - self.entropy_coeff * entropy
                + self.kl_coef * kl_t
            )

            self.policy_optim.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.p_agent.parameters(), 1.0)
            self.policy_optim.step()

            approx_kl = float(approx_kl_t.detach().item())
            true_kl = float(kl_t.detach().item())
            final_policy_loss = policy_loss.detach()
            final_value_loss = value_loss.detach()
            final_entropy = entropy.detach()
            final_total_loss = total_loss.detach()

        # Adaptive KL update
        if true_kl > self.target_kl:
            self.kl_coef *= 1.5
        elif true_kl < self.target_kl / 2:
            self.kl_coef *= 0.8

        metrics = {
            'policy_loss': float(final_policy_loss.item()),
            'value_loss': float(final_value_loss.item()),
            'entropy': float(final_entropy.item()),
            'total_loss': float(final_total_loss.item()),
            'kl_coef': float(self.kl_coef),
            'kl': true_kl,
        }
        metrics['approx_kl'] = approx_kl
        metrics['policy_entropy'] = metrics.get('entropy', 0.0)
        return metrics

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
        requested_action = 2  # default to REGENERATE at episode start
        for step in range(max_steps):
            if requested_action == 3:  # STOP action from previous step
                break

            smiles, reward, scores, transition, selected_action = self._step(
                prev_reward=prev_reward,
                requested_action=requested_action,
            )
            transitions.append(transition)
            episode_smiles.append(smiles)
            episode_rewards.append(reward)
            episode_yields.append(scores.get('yield', 0.0))
            episode_vis_var.append(scores.get('vision_var', 0.0))
            episode_sigma.append(scores.get('yield_sigma', 0.0))
            prev_reward = reward
            requested_action = selected_action

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

            if selected_action == 3:  # STOP immediately ends episode
                break

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
        invalid_count = max_steps - valid_count
        pct_invalid = 100.0 - pct_valid
        avg_reward   = sum(episode_rewards) / max(1, len(episode_rewards))
        diversity    = diversity_score(episode_smiles)

        metrics = {
            'episode':            episode_idx,
            'avg_reward':         avg_reward,
            'best_reward':        self.best_reward,
            'pct_valid':          pct_valid,
            'pct_invalid':        pct_invalid,
            'invalid_count':      invalid_count,
            'diversity':          diversity,
            'vision_variance':    avg_vis_var,
            'yield_sa_corr':      yield_sa_corr,
            'yield_uncertainty':  avg_yield_sigma,
            **loss_info,
        }

        logger.info(
            f"Episode {episode_idx:3d} | avg_rwd={avg_reward:.3f} | "
            f"best={self.best_reward:.3f} | valid={pct_valid:.0f}% | invalid={pct_invalid:.0f}% | "
            f"div={diversity:.3f} | kl={loss_info.get('approx_kl', 0):.4f} | "
            f"ent={loss_info.get('policy_entropy', 0):.4f} | loss={loss_info.get('total_loss', 0):.4f} | "
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
        self._plot_validity_curve(save_csv)
        logger.info(
            f"\n{'='*60}\nTraining complete!\n"
            f"Best reward : {self.best_reward:.4f}\n"
            f"Best SMILES : {self.best_smiles or '<none>'}\n"
            f"{'='*60}"
        )

    def _plot_validity_curve(self, metrics_csv: str) -> None:
        """Plot validity-over-time from the episode metrics CSV."""
        try:
            import pandas as pd
            import matplotlib.pyplot as plt
        except Exception as exc:
            logger.warning(f"Skipping validity plot (missing deps): {exc}")
            return

        try:
            df = pd.read_csv(metrics_csv)
            if 'episode' not in df.columns or 'pct_valid' not in df.columns:
                logger.warning(f"Skipping validity plot (missing columns): {metrics_csv}")
                return
            plt.figure(figsize=(8, 4))
            plt.plot(df['episode'], df['pct_valid'], label='% Valid SMILES', color='seagreen', linewidth=2)
            plt.ylim(0, 100)
            plt.xlabel('Episode')
            plt.ylabel('% Valid')
            plt.title('SMILES Validity Over Time')
            plt.grid(alpha=0.3)
            plt.legend()
            out_png = os.path.splitext(metrics_csv)[0] + '_validity.png'
            plt.tight_layout()
            plt.savefig(out_png, dpi=150)
            plt.close()
            logger.info(f"Validity plot saved → {out_png}")
        except Exception as exc:
            logger.warning(f"Failed to plot validity curve: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='MATMED Training')
    p.add_argument('--num_episodes',     type=int,   default=20)
    p.add_argument('--steps_per_episode',type=int,   default=64)
    p.add_argument('--num_pretrain',     type=int,   default=5)
    p.add_argument('--lr',               type=float, default=1e-5)
    p.add_argument('--gamma',            type=float, default=0.99)
    p.add_argument('--entropy_coeff',    type=float, default=0.1)
    p.add_argument('--ppo_clip',         type=float, default=0.1)
    p.add_argument('--ppo_epochs',       type=int,   default=4)
    p.add_argument('--kl_coef',          type=float, default=0.1)
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
        ppo_clip=args.ppo_clip,
        kl_coef=args.kl_coef,
        ppo_epochs=args.ppo_epochs,
        seed=args.seed,
        use_chemberta=args.use_chemberta,
    )

    runner.train(
        num_episodes=args.num_episodes,
        steps_per_episode=args.steps_per_episode,
        save_csv=args.save_csv,
    )

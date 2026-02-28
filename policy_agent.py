"""
policy_agent.py — P-Agent: Policy Transformer
===============================================
The core novelty of MATMED: a cross-agent policy that reads embeddings from
all four specialist agents and decides what to do next.

Scientific rationale:
  Single-objective optimisation of drug candidates leads to solutions that
  score well on one metric while failing on others (e.g., high binding
  affinity but poor ADMET).  By jointly processing embeddings from the
  Generator, Evaluator, Safety, and Reaction agents through a Transformer
  encoder, the policy learns a *multi-objective* decision function that
  integrates all signals before committing to an action.

Inputs (per step):
  - g_emb     : (d_g,) embedding from G-Agent
  - e_emb     : (d_e,) embedding from E-Agent
  - s_emb     : (d_s,) embedding from S-Agent
  - r_emb     : (d_r,) embedding from R-Agent
  - reward    : scalar reward from the previous step (context)

Processing:
  1. Project each embedding to a common d_model dimension.
  2. Prepend a learned [CLS] token.
  3. Stack as a sequence of 5 tokens.
  4. Pass through a 2-layer Transformer encoder.
  5. Extract CLS token → MLP decision head → 4-class logits.

Actions:
  0 = ACCEPT      — current molecule is good enough, record it
  1 = MODIFY      — request targeted modification from G-Agent
  2 = REGENERATE  — discard and generate from scratch
  3 = STOP        — terminate the optimisation episode

Reinforcement learning:
  Simple REINFORCE (Williams, 1992) with a baseline:
    ∇θ J ≈ (R_t - b) ∇θ log π(a_t | s_t)

  PPO-style clipping is also supported for more stable training.
"""

import math
from typing import Optional, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import get_logger, set_seed

logger = get_logger('P-Agent')


# ─────────────────────────────────────────────────────────────────────────────
# Action space
# ─────────────────────────────────────────────────────────────────────────────

ACTION_ACCEPT      = 0
ACTION_MODIFY      = 1
ACTION_REGENERATE  = 2
ACTION_STOP        = 3
ACTION_NAMES       = ['ACCEPT', 'MODIFY', 'REGENERATE', 'STOP']
NUM_ACTIONS        = 4


# ─────────────────────────────────────────────────────────────────────────────
# Policy Agent
# ─────────────────────────────────────────────────────────────────────────────

class PolicyAgent(nn.Module):
    """
    P-Agent: Multi-objective Policy Transformer.

    Reads latent embeddings from all specialist agents and outputs a
    probability distribution over discrete actions.

    Attributes:
        d_g, d_e, d_s, d_r: Input embedding dimensions for each agent.
        d_model:            Hidden dimension of the policy Transformer.
        nhead:              Attention heads (default 4).
        num_layers:         Transformer encoder layers (default 2).
    """

    def __init__(
        self,
        d_g: int = 256,   # G-Agent embedding dim
        d_e: int = 128,   # E-Agent embedding dim
        d_s: int = 128,   # S-Agent embedding dim
        d_r: int = 128,   # R-Agent embedding dim
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        # Per-agent linear projections → d_model
        self.proj_g = nn.Linear(d_g, d_model)
        self.proj_e = nn.Linear(d_e, d_model)
        self.proj_s = nn.Linear(d_s, d_model)
        self.proj_r = nn.Linear(d_r, d_model)
        self.proj_reward = nn.Linear(1, d_model)   # scalar reward as context

        # Learned [CLS] token embedding
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Learned positional embeddings for a fixed-length sequence
        # Sequence: [CLS, g, e, s, r, reward] = 6 tokens
        self.pos_embed = nn.Embedding(7, d_model)

        # Transformer encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.ln = nn.LayerNorm(d_model)

        # MLP decision head: d_model → NUM_ACTIONS
        self.decision_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, NUM_ACTIONS),
        )

        # Value head for advantage estimation (baseline)
        self.value_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.decision_head[-1].weight)
        nn.init.zeros_(self.decision_head[-1].bias)
        nn.init.xavier_uniform_(self.value_head[-1].weight)
        nn.init.zeros_(self.value_head[-1].bias)

    def forward(
        self,
        g_emb: torch.Tensor,
        e_emb: torch.Tensor,
        s_emb: torch.Tensor,
        r_emb: torch.Tensor,
        reward: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute action logits and state value from agent embeddings.

        Args:
            g_emb:  (d_g,) or (B, d_g) G-Agent embedding.
            e_emb:  (d_e,) or (B, d_e) E-Agent embedding.
            s_emb:  (d_s,) or (B, d_s) S-Agent embedding.
            r_emb:  (d_r,) or (B, d_r) R-Agent embedding.
            reward: Scalar reward signal from previous step.

        Returns:
            action_logits: (B, NUM_ACTIONS) raw action scores.
            action_probs:  (B, NUM_ACTIONS) probability distribution.
            value:         (B, 1) estimated state value.
        """
        # Ensure batch dimension
        if g_emb.dim() == 1:
            g_emb = g_emb.unsqueeze(0)
            e_emb = e_emb.unsqueeze(0)
            s_emb = s_emb.unsqueeze(0)
            r_emb = r_emb.unsqueeze(0)

        B = g_emb.size(0)
        device = g_emb.device

        # Project each embedding to d_model → (B, 1, d_model)
        g_tok = self.proj_g(g_emb).unsqueeze(1)
        e_tok = self.proj_e(e_emb).unsqueeze(1)
        s_tok = self.proj_s(s_emb).unsqueeze(1)
        r_tok = self.proj_r(r_emb).unsqueeze(1)
        rew_tok = self.proj_reward(
            torch.tensor([[reward]], dtype=torch.float, device=device).expand(B, 1)
        ).unsqueeze(1)

        # CLS token
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)

        # Build token sequence: [CLS, g, e, s, r, reward]
        seq = torch.cat([cls, g_tok, e_tok, s_tok, r_tok, rew_tok], dim=1)  # (B, 6, d_model)

        # Add positional embeddings
        pos_ids = torch.arange(seq.size(1), device=device)
        seq = seq + self.pos_embed(pos_ids).unsqueeze(0)

        # Transformer
        h = self.transformer(seq)              # (B, 6, d_model)
        cls_out = self.ln(h[:, 0, :])         # (B, d_model)

        action_logits = self.decision_head(cls_out)       # (B, NUM_ACTIONS)
        action_probs  = F.softmax(action_logits, dim=-1)
        value         = self.value_head(cls_out)           # (B, 1)

        return action_logits, action_probs, value

    @torch.no_grad()
    def select_action(
        self,
        g_emb: torch.Tensor,
        e_emb: torch.Tensor,
        s_emb: torch.Tensor,
        r_emb: torch.Tensor,
        reward: float = 0.0,
        greedy: bool = False,
    ) -> Tuple[int, float, torch.Tensor]:
        """
        Sample (or greedily select) an action given current state.

        Args:
            g_emb, e_emb, s_emb, r_emb: Agent embeddings.
            reward: Previous step reward.
            greedy: If True, select argmax action; otherwise sample.

        Returns:
            action:     Integer action index.
            log_prob:   Log-probability of the selected action.
            value:      Scalar value estimate.
        """
        _, probs, value = self.forward(g_emb, e_emb, s_emb, r_emb, reward)
        if greedy:
            action = int(probs.argmax(dim=-1).item())
        else:
            dist  = torch.distributions.Categorical(probs)
            action = int(dist.sample().item())

        log_prob = float(torch.log(probs[0, action] + 1e-8).item())
        return action, log_prob, value.squeeze()

    def compute_policy_loss(
        self,
        log_probs: torch.Tensor,
        rewards: torch.Tensor,
        values: torch.Tensor,
        gamma: float = 0.99,
        entropy_coeff: float = 0.01,
        action_probs_all: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute REINFORCE loss with value-function baseline (advantage).

        Args:
            log_probs:        (T,) log-probabilities of taken actions.
            rewards:          (T,) actual rewards received.
            values:           (T,) value estimates V(s_t).
            gamma:            Discount factor.
            entropy_coeff:    Coefficient for entropy regularisation bonus.
            action_probs_all: (T, NUM_ACTIONS) for entropy; optional.

        Returns:
            Dict with 'policy_loss', 'value_loss', 'entropy', 'total_loss'.
        """
        # Compute discounted returns
        returns = _discount_returns(rewards, gamma)

        # Advantages
        advantages = returns - values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Policy gradient loss
        policy_loss = -(log_probs * advantages).mean()

        # Value function loss (MSE)
        value_loss = F.mse_loss(values, returns)

        # Entropy bonus (only if probs provided)
        entropy = torch.tensor(0.0)
        if action_probs_all is not None:
            entropy = -(action_probs_all * torch.log(action_probs_all + 1e-8)).sum(-1).mean()

        total_loss = policy_loss + 0.5 * value_loss - entropy_coeff * entropy

        return {
            'policy_loss': policy_loss,
            'value_loss':  value_loss,
            'entropy':     entropy,
            'total_loss':  total_loss,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def _discount_returns(rewards: torch.Tensor, gamma: float = 0.99) -> torch.Tensor:
    """
    Compute discounted cumulative returns G_t = Σ_{k≥0} γ^k R_{t+k}.

    Args:
        rewards: (T,) reward tensor.
        gamma:   Discount factor.

    Returns:
        (T,) return tensor.
    """
    T = rewards.size(0)
    returns = torch.zeros_like(rewards)
    G = 0.0
    for t in reversed(range(T)):
        G = rewards[t].item() + gamma * G
        returns[t] = G
    return returns


# ─────────────────────────────────────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    set_seed(42)

    agent = PolicyAgent()
    print(f"P-Agent parameters: {sum(p.numel() for p in agent.parameters()):,}")

    # Fake embeddings
    g = torch.randn(256)
    e = torch.randn(128)
    s = torch.randn(128)
    r = torch.randn(128)

    logits, probs, val = agent.forward(g, e, s, r, reward=0.5)
    print(f"Action probs : {probs.squeeze().detach().numpy().round(4)}")
    print(f"Value est.   : {val.item():.4f}")
    print(f"Action names : {ACTION_NAMES}")

    action, lp, v = agent.select_action(g, e, s, r, reward=0.5)
    print(f"Selected action: {ACTION_NAMES[action]}  (log_prob={lp:.4f})")

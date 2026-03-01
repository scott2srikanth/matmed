"""
reaction_agent.py — R-Agent: Research-Grade Reaction Predictor (Phase 3 v2)
============================================================================
Implements Parts II, III, IV of the MATMED research spec:

  Part II:  Cross-Attention Fusion
            - Project m → m' via W_m before cross-attention (as per spec)
            - m' is Query; h_vision is Key and Value
  Part III: Yield Prediction Head (linear → bounded yield ∈ [0,1])
  Part IV:  MC Dropout Uncertainty Estimation
            - K stochastic forward passes at inference
            - Mean μ and variance σ² over yields
            - Returns (μ, σ²) for uncertainty-penalized reward R = μ - λσ
"""

import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import Descriptors

from utils import get_logger, set_seed, is_valid_smiles

logger = get_logger('R-Agent')


# ─────────────────────────────────────────────────────────────────────────────
# Descriptor computation
# ─────────────────────────────────────────────────────────────────────────────

def _compute_sa_score(mol) -> float:
    try:
        from rdkit.Contrib.SA_Score import sascorer      # type: ignore
        return float(sascorer.calculateScore(mol))
    except Exception:
        pass
    try:
        return min(10.0, 1.0 + mol.GetNumHeavyAtoms() / 10.0)
    except Exception:
        return 5.0


def compute_reaction_features(smiles: str) -> Optional[np.ndarray]:
    """
    (3,) → [norm_sa, norm_mw, norm_lp].  Returns None if SMILES is invalid.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    sa   = _compute_sa_score(mol)
    mw   = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    return np.array([
        1.0 - (sa - 1.0) / 9.0,              # norm_sa
        math.exp(-mw / 500.0),                # norm_mw
        math.exp(-((logp - 2.5) ** 2) / 4.0), # norm_lp
    ], dtype=np.float32)


REACTION_FEAT_DIM = 3


# ─────────────────────────────────────────────────────────────────────────────
# Part II — Cross-Attention Fusion with W_m Molecular Projection
# ─────────────────────────────────────────────────────────────────────────────

class CrossAttentionFusion(nn.Module):
    """
    Spec Part II — fuses m' (projected molecule) with h_vision via cross-attention.

        m'      = W_m m                   (project molecule into d_v space)
        Q       = m' W_Q^c
        K, V    = h_vision W_K^c, W_V^c
        Attn    = softmax(QK^T / sqrt(d)) V
        h_fused = LayerNorm(m' + Attn)
    """

    def __init__(
        self,
        mol_dim: int = 128,        # d_m — input molecule embedding dim
        vis_dim: int = 128,        # d_v — vision embedding dim
        proj_dim: int = 128,       # projected common dim d
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        # W_m: project molecule m → m'
        self.mol_proj = nn.Linear(mol_dim, proj_dim)

        # Multi-head cross-attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=proj_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
            kdim=vis_dim,
            vdim=vis_dim,
        )
        self.norm = nn.LayerNorm(proj_dim)

    def forward(
        self,
        mol_emb: torch.Tensor,   # (B, mol_dim)
        vis_emb: torch.Tensor,   # (B, vis_dim)
    ) -> torch.Tensor:           # (B, proj_dim)
        m_prime = self.mol_proj(mol_emb)            # W_m m  → (B, proj_dim)
        q  = m_prime.unsqueeze(1)                   # (B, 1, proj_dim)
        kv = vis_emb.unsqueeze(1)                   # (B, 1, vis_dim)

        attn_out, _ = self.cross_attn(
            query=q, key=kv, value=kv
        )                                           # (B, 1, proj_dim)

        h_fused = self.norm(m_prime + attn_out.squeeze(1))  # residual + LN
        return h_fused


# ─────────────────────────────────────────────────────────────────────────────
# Part IV — MC Dropout for uncertainty estimation
# ─────────────────────────────────────────────────────────────────────────────

def mc_dropout_predict(
    head: nn.Module,
    h: torch.Tensor,
    K: int = 10,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Spec Part IV: K stochastic forward passes with dropout enabled.

        μ  = mean  of {ŷ_1, ..., ŷ_K}
        σ² = var   of {ŷ_1, ..., ŷ_K}

    Args:
        head: The yield prediction head (must contain Dropout layers).
        h:    (B, dim) fused embedding.
        K:    Number of stochastic passes.

    Returns:
        mu:  (B,)  mean yield prediction.
        var: (B,)  variance (uncertainty).
    """
    head.train()   # keep dropout active
    preds = torch.stack([head(h).squeeze(-1) for _ in range(K)], dim=0)  # (K, B)
    mu  = preds.mean(dim=0)   # (B,)
    var = preds.var(dim=0)    # (B,)
    return mu, var


# ─────────────────────────────────────────────────────────────────────────────
# R-Agent
# ─────────────────────────────────────────────────────────────────────────────

class ReactionAgent(nn.Module):
    """
    R-Agent (Research Grade v2):

      - Molecular encoder: 3-dim descriptors → mol_emb ∈ R^{hidden_dim}
      - Vision path:       (T, d_raw=28) → VisionTemporalTransformer → h_vision ∈ R^{128}
      - Fusion:            CrossAttentionFusion(mol_emb, h_vision) → h_fused
      - Yield head:        h_fused → ŷ ∈ [0,1]   (Part III)
      - Uncertainty:       MC Dropout K passes → (μ, σ²)            (Part IV)
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        dropout: float = 0.15,    # slightly higher dropout for MC sampling
        use_vision: bool = True,
        vision_input_dim: int = 28,   # d_raw = 28 (physics-based features)
        vision_embed_dim: int = 128,  # d_v  = 128
        vision_layers: int = 3,
        vision_heads: int = 4,
        mc_samples: int = 10,         # K in spec
        uncertainty_lambda: float = 0.1,  # λ in R = μ - λσ
    ) -> None:
        super().__init__()
        self.hidden_dim  = hidden_dim
        self.use_vision  = use_vision
        self.mc_samples  = mc_samples
        self.lam         = uncertainty_lambda

        # ── Molecular encoder ────────────────────────────────────────────────
        self.mol_encoder = nn.Sequential(
            nn.Linear(REACTION_FEAT_DIM, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, hidden_dim),
            nn.GELU(),
        )
        self.mol_ln = nn.LayerNorm(hidden_dim)

        # ── Vision transformer ───────────────────────────────────────────────
        if use_vision:
            from vision_agent import VisionTemporalTransformer
            self.vision_transformer = VisionTemporalTransformer(
                input_dim=vision_input_dim,
                embed_dim=vision_embed_dim,
                num_layers=vision_layers,
                num_heads=vision_heads,
            )
            # Part II: W_m projection + cross-attention
            self.cross_attn = CrossAttentionFusion(
                mol_dim=hidden_dim,
                vis_dim=vision_embed_dim,
                proj_dim=hidden_dim,
                num_heads=4,
            )
        else:
            self.vision_transformer = None
            self.cross_attn = None

        # ── Part III: Yield prediction head (with dropout for MC sampling) ───
        self.yield_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),     # dropout kept active for MC sampling
            nn.Linear(64, 1),
            nn.Sigmoid(),            # bound output to [0, 1]
        )

    # ── Internal feature → yield ─────────────────────────────────────────────

    def _encode_mol(self, feat: torch.Tensor) -> torch.Tensor:
        return self.mol_ln(self.mol_encoder(feat))   # (B, hidden_dim)

    def forward_features(
        self,
        feat: torch.Tensor,                          # (B, 3)
        vision_seq: Optional[torch.Tensor] = None,   # (B, T, d_raw)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns: (mu, sigma_sq, h_fused)
        """
        mol_emb = self._encode_mol(feat)   # (B, D)

        if self.use_vision and vision_seq is not None and self.vision_transformer is not None:
            h_vision = self.vision_transformer(vision_seq)   # (B, d_v)
            h_fused  = self.cross_attn(mol_emb, h_vision)    # (B, D) — Part II
        else:
            h_fused = mol_emb

        # Part IV: MC Dropout uncertainty
        mu, var = mc_dropout_predict(self.yield_head, h_fused, K=self.mc_samples)
        return mu, var, h_fused

    # ── Public forward: single SMILES ────────────────────────────────────────

    def forward(
        self,
        smiles: str,
        vision_seq: Optional[torch.Tensor] = None,   # (T, d_raw) or (1, T, d_raw)
    ) -> Tuple[float, torch.Tensor]:
        """
        Returns:
            yield_score: float   μ - λσ  (uncertainty-penalized yield)
            embedding:   (hidden_dim,) h_fused tensor for P-Agent
        """
        feats  = compute_reaction_features(smiles)
        device = next(self.parameters()).device

        if feats is None:
            return 0.0, torch.zeros(self.hidden_dim, device=device)

        t_feat = torch.tensor(feats, dtype=torch.float, device=device).unsqueeze(0)

        v_seq = None
        if vision_seq is not None:
            v_seq = (vision_seq.unsqueeze(0) if vision_seq.dim() == 2 else vision_seq).to(device)

        mu, var, h_fused = self.forward_features(t_feat, vision_seq=v_seq)

        sigma = torch.sqrt(var.clamp(min=1e-8))
        # R = μ - λσ  (exploration-aware, spec Part IV)
        score = float((mu[0] - self.lam * sigma[0]).item())
        score = max(0.0, min(1.0, score))   # clamp to [0, 1]

        return score, h_fused[0].detach()

    def get_uncertainty(
        self,
        smiles: str,
        vision_seq: Optional[torch.Tensor] = None,
    ) -> Tuple[float, float]:
        """Convenience method: returns (mu, sigma) for logging."""
        feats  = compute_reaction_features(smiles)
        device = next(self.parameters()).device
        if feats is None:
            return 0.0, 0.0
        t_feat = torch.tensor(feats, dtype=torch.float, device=device).unsqueeze(0)
        v_seq  = None
        if vision_seq is not None:
            v_seq = (vision_seq.unsqueeze(0) if vision_seq.dim() == 2 else vision_seq).to(device)
        mu, var, _ = self.forward_features(t_feat, vision_seq=v_seq)
        return float(mu[0].item()), float(torch.sqrt(var[0].clamp(min=1e-8)).item())


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    from utils import get_zinc_sample
    from vision_agent import simulate_reaction_video
    set_seed(42)

    agent = ReactionAgent(use_vision=True, mc_samples=10)
    print(f'R-Agent (v2) parameters: {sum(p.numel() for p in agent.parameters()):,}')

    for smi in get_zinc_sample()[:3]:
        feats = compute_reaction_features(smi)
        if feats is not None:
            vis_seq = simulate_reaction_video(feats)   # (50, 28!)
            score, emb = agent(smi, vision_seq=vis_seq)
            mu, sigma  = agent.get_uncertainty(smi, vision_seq=vis_seq)
        else:
            score, emb = agent(smi)
            mu, sigma  = 0.0, 0.0
        print(
            f'  yield={score:.4f}  μ={mu:.4f}  σ={sigma:.4f}  '
            f'emb={emb.shape}  smi={smi[:35]}'
        )
    print('\n✅ Research-grade R-Agent self-test passed!')

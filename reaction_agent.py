"""
reaction_agent.py — R-Agent: Reaction Feasibility Predictor (Phase 3)
======================================================================
Phase 3 Upgrade: Now integrates vision embeddings from VisionTemporalTransformer
via cross-attention to improve yield prediction.

Original heuristic yield:
  raw_yield = w_sa   * (1 - normalised(SA_score))
            + w_mw   * heaviside(MolWt, 500)
            + w_logP * gaussian(logP, centre=2.5)

Phase 3 Enhancement:
  The molecular embedding from the MLP encoder attends (as Query) to the
  vision embedding from VisionTemporalTransformer (as Key and Value).
  This allows the yield predictor to incorporate dynamic, time-dependent signals
  alongside the static physicochemical descriptors.

Output:
  - yield_score : float ∈ [0, 1]
  - embedding   : (hidden_dim,) tensor
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
# RDKit descriptor computation
# ─────────────────────────────────────────────────────────────────────────────

def _compute_sa_score(mol) -> float:
    """Compute Synthetic Accessibility score; fallback to neutral 5.0."""
    try:
        from rdkit.Contrib.SA_Score import sascorer      # type: ignore
        return float(sascorer.calculateScore(mol))
    except Exception:
        pass
    try:
        from rdkit.Chem import rdMolDescriptors
        return min(10.0, 1.0 + mol.GetNumHeavyAtoms() / 10.0)
    except Exception:
        return 5.0


def compute_reaction_features(smiles: str) -> Optional[np.ndarray]:
    """
    Compute a 3-dimensional feature vector from physicochemical descriptors.

    Features (all normalised to approximately [0, 1]):
      [0] norm_sa : 1 - (SA_score - 1) / 9
      [1] norm_mw : exp(-MolWt / 500)
      [2] norm_lp : exp(-(logP - 2.5)^2 / 4)

    Returns:
        (3,) numpy array or None if SMILES is invalid.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    sa   = _compute_sa_score(mol)
    mw   = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)

    norm_sa = 1.0 - (sa - 1.0) / 9.0
    norm_mw = math.exp(-mw / 500.0)
    norm_lp = math.exp(-((logp - 2.5) ** 2) / 4.0)

    return np.array([norm_sa, norm_mw, norm_lp], dtype=np.float32)


REACTION_FEAT_DIM = 3


# ─────────────────────────────────────────────────────────────────────────────
# Cross-Attention Fusion Module
# ─────────────────────────────────────────────────────────────────────────────

class CrossAttentionFusion(nn.Module):
    """
    Fuses a molecular embedding (query) with a vision embedding (key/value)
    via multi-head cross-attention, producing an enriched fused embedding.
    """

    def __init__(self, embed_dim: int = 128, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        mol_emb: torch.Tensor,      # (B, D)
        vis_emb: torch.Tensor,      # (B, D)
    ) -> torch.Tensor:              # (B, D)
        # Unsqueeze to add sequence dimension: (B, 1, D)
        q = mol_emb.unsqueeze(1)
        kv = vis_emb.unsqueeze(1)

        # Cross-attend: molecule queries vision
        attended, _ = self.attn(query=q, key=kv, value=kv)

        # Residual connection + LayerNorm
        fused = self.norm(mol_emb + attended.squeeze(1))
        return fused


# ─────────────────────────────────────────────────────────────────────────────
# R-Agent (Phase 3 with Vision)
# ─────────────────────────────────────────────────────────────────────────────

class ReactionAgent(nn.Module):
    """
    R-Agent (Phase 3): Predicts synthetic yield score from molecular descriptors
    optionally enriched with vision embeddings from VisionTemporalTransformer.

    Usage (with vision)::
        agent = ReactionAgent()
        vision_seq = simulate_reaction_video(feats)           # (50, 16)
        yield_score, emb = agent("CC(=O)Oc1ccccc1C(=O)O", vision_seq.unsqueeze(0))

    Usage (without vision, backward compatible)::
        yield_score, emb = agent("CC(=O)Oc1ccccc1C(=O)O")
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        use_vision: bool = True,
        vision_input_dim: int = 16,
        vision_embed_dim: int = 128,
        vision_seq_len: int = 50,
        vision_layers: int = 3,
        vision_heads: int = 4,
    ) -> None:
        super().__init__()
        self.hidden_dim  = hidden_dim
        self.use_vision  = use_vision

        # ── Molecular feature encoder: 3 → hidden_dim ──────────────────────
        self.encoder = nn.Sequential(
            nn.Linear(REACTION_FEAT_DIM, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, hidden_dim),
            nn.GELU(),
        )
        self.mol_ln = nn.LayerNorm(hidden_dim)

        # ── Vision transformer (imported lazily to avoid circular imports) ──
        if use_vision:
            from vision_agent import VisionTemporalTransformer
            self.vision_transformer = VisionTemporalTransformer(
                input_dim=vision_input_dim,
                embed_dim=vision_embed_dim,
                num_layers=vision_layers,
                num_heads=vision_heads,
            )
            self.cross_attn = CrossAttentionFusion(
                embed_dim=hidden_dim,
                num_heads=4,
            )
        else:
            self.vision_transformer = None
            self.cross_attn = None

        # ── Yield prediction head ───────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward_features(
        self,
        feat: torch.Tensor,                         # (B, 3)
        vision_seq: Optional[torch.Tensor] = None,  # (B, T, 16)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            feat:       (batch, 3) descriptor tensor.
            vision_seq: (batch, seq_len, feature_dim) optional vision sequence.

        Returns:
            yield_scores: (batch,) tensor.
            embeddings:   (batch, hidden_dim) tensor.
        """
        mol_emb = self.mol_ln(self.encoder(feat))   # (B, D)

        if self.use_vision and vision_seq is not None and self.vision_transformer is not None:
            vis_emb = self.vision_transformer(vision_seq)   # (B, D)
            fused   = self.cross_attn(mol_emb, vis_emb)     # (B, D)
        else:
            fused = mol_emb

        score = self.head(fused).squeeze(-1)
        return score, fused

    def forward(
        self,
        smiles: str,
        vision_seq: Optional[torch.Tensor] = None,   # (1, T, 16) or (T, 16)
    ) -> Tuple[float, torch.Tensor]:
        """
        Predict yield score for a single SMILES string, optionally using vision.

        Args:
            smiles:     SMILES string.
            vision_seq: (1, seq_len, 16) or (seq_len, 16) vision tensor (optional).

        Returns:
            yield_score: float ∈ [0, 1].
            embedding:   (hidden_dim,) tensor.
        """
        feats = compute_reaction_features(smiles)
        device = next(self.parameters()).device

        if feats is None:
            return 0.0, torch.zeros(self.hidden_dim, device=device)

        t_feat = torch.tensor(feats, dtype=torch.float, device=device).unsqueeze(0)  # (1, 3)

        # Handle optional vision sequence
        v_seq = None
        if vision_seq is not None:
            if vision_seq.dim() == 2:           # (T, D) -> (1, T, D)
                vision_seq = vision_seq.unsqueeze(0)
            v_seq = vision_seq.to(device)

        scores, embs = self.forward_features(t_feat, vision_seq=v_seq)
        return float(scores[0].item()), embs[0]


# ─────────────────────────────────────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    from utils import get_zinc_sample
    from vision_agent import simulate_reaction_video
    set_seed(42)

    agent = ReactionAgent(use_vision=True)
    print(f"R-Agent parameters: {sum(p.numel() for p in agent.parameters()):,}")

    for smi in get_zinc_sample()[:3]:
        # Generate vision sequence for this molecule
        feats = compute_reaction_features(smi)
        if feats is not None:
            vis_seq = simulate_reaction_video(feats)   # (50, 16)
            vision_var = float(torch.var(vis_seq).item())
            score, emb = agent(smi, vision_seq=vis_seq)
        else:
            vision_var = 0.0
            score, emb = agent(smi)
        print(
            f"  yield={score:.4f}  vision_var={vision_var:.4f}  "
            f"emb={emb.shape}  smi={smi[:40]}"
        )
    print("✅ R-Agent (Phase 3 Vision) self-test passed!")

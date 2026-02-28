"""
reaction_agent.py — R-Agent: Reaction Feasibility Predictor
=============================================================
Predicts the synthetic feasibility (yield score) of a candidate molecule.

Scientific rationale:
  A molecule may bind a target strongly but be practically unsynthesisable.
  The Synthetic Accessibility (SA) score (Ertl & Schuffenhauer, 2009)
  quantifies how easily a compound can be made in a lab.  Combined with
  molecular weight and logP, these physicochemical descriptors form a
  heuristic proxy for reaction yield — sufficient for an RL prototype where
  no actual lab data is available.

Heuristic yield formula:
  raw_yield = w_sa   * (1 - normalised(SA_score))   # SA in [1,10] → invert
            + w_mw   * heaviside(MolWt, 500)          # penalise heavy molecules
            + w_logP * gaussian(logP, centre=2.5)      # prefer logP ≈ 2.5

These raw features are passed through a small MLP (optionally a tiny
Transformer) to allow learned nonlinear combinations.

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

# SA_score sub-module (rdkit contrib) — graceful fallback
def _compute_sa_score(mol) -> float:
    """Compute Synthetic Accessibility score; fallback to neutral 5.0."""
    try:
        from rdkit.Contrib.SA_Score import sascorer      # type: ignore
        return float(sascorer.calculateScore(mol))
    except Exception:
        pass
    try:
        from rdkit.Chem import rdMolDescriptors
        # Rough proxy: use HeavyAtomCount normalised
        return min(10.0, 1.0 + mol.GetNumHeavyAtoms() / 10.0)
    except Exception:
        return 5.0


def compute_reaction_features(smiles: str) -> Optional[np.ndarray]:
    """
    Compute a 3-dimensional feature vector from physicochemical descriptors.

    Features (all normalised to approximately [0, 1]):
      [0] norm_sa : 1 - (SA_score - 1) / 9   (higher = easier synthesis)
      [1] norm_mw : exp(-MolWt / 500)          (higher = lighter molecule)
      [2] norm_lp : exp(-(logP - 2.5)^2 / 4)  (higher = closer to ideal logP)

    Args:
        smiles: Valid SMILES string.

    Returns:
        (3,) numpy array or None if SMILES is invalid.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    sa   = _compute_sa_score(mol)
    mw   = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)

    norm_sa = 1.0 - (sa - 1.0) / 9.0          # SA ∈ [1,10] → [0,1]
    norm_mw = math.exp(-mw / 500.0)            # lighter → higher score
    norm_lp = math.exp(-((logp - 2.5) ** 2) / 4.0)  # Lipophilicity preference

    return np.array([norm_sa, norm_mw, norm_lp], dtype=np.float32)


REACTION_FEAT_DIM = 3


# ─────────────────────────────────────────────────────────────────────────────
# MLP Reaction Agent
# ─────────────────────────────────────────────────────────────────────────────

class ReactionAgent(nn.Module):
    """
    R-Agent: Predicts synthetic yield score from molecular descriptors.

    The 3-dim descriptor vector (SA, MolWt, logP) is passed through a small
    MLP that maps it to a hidden embedding and a scalar yield score.

    For the prototype the MLP replaces a full Transformer to keep compute
    low, while still matching the spirit of the design (a learnable module
    that can be supervised with real yield labels if available).

    Usage::

        agent = ReactionAgent()
        yield_score, emb = agent("CC(=O)Oc1ccccc1C(=O)O")
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        # Feature encoder: 3 → hidden_dim
        self.encoder = nn.Sequential(
            nn.Linear(REACTION_FEAT_DIM, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, hidden_dim),
            nn.GELU(),
        )

        self.ln = nn.LayerNorm(hidden_dim)

        # Yield prediction head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),   # output ∈ [0, 1]
        )

    def forward_features(self, feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            feat: (batch, 3) descriptor tensor.

        Returns:
            yield_scores: (batch,) tensor.
            embeddings:   (batch, hidden_dim) tensor.
        """
        emb   = self.ln(self.encoder(feat))
        score = self.head(emb).squeeze(-1)
        return score, emb

    def forward(self, smiles: str) -> Tuple[float, torch.Tensor]:
        """
        Predict yield score for a single SMILES string.

        Args:
            smiles: SMILES string.

        Returns:
            yield_score: float ∈ [0, 1].
            embedding:   (hidden_dim,) tensor — passed to P-Agent.
        """
        feats = compute_reaction_features(smiles)
        if feats is None:
            device = next(self.parameters()).device
            zero_emb = torch.zeros(self.hidden_dim, device=device)
            return 0.0, zero_emb

        device  = next(self.parameters()).device
        t_feat  = torch.tensor(feats, dtype=torch.float, device=device).unsqueeze(0)
        scores, embs = self.forward_features(t_feat)
        return float(scores[0].item()), embs[0]


# ─────────────────────────────────────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    from utils import get_zinc_sample
    set_seed(42)

    agent = ReactionAgent()
    print(f"R-Agent parameters: {sum(p.numel() for p in agent.parameters()):,}")

    for smi in get_zinc_sample()[:5]:
        score, emb = agent(smi)
        feats = compute_reaction_features(smi)
        print(
            f"  yield={score:.4f}  feats={np.round(feats, 3)}  "
            f"emb_shape={emb.shape}  smi={smi[:40]}"
        )

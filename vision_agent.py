"""
vision_agent.py — Phase 3: Research-Grade Vision-Agent (v2)
============================================================
Implements the exact mathematical formulation from the MATMED research spec:

  Part I  — VisionTemporalTransformer (attention pooling, physics-based features)
  Part II — Cross-attention fusion with W_m molecular projection  →  see reaction_agent.py
  Part III— Yield prediction head                                 →  see reaction_agent.py
  Part IV — MC Dropout uncertainty estimation                     →  see reaction_agent.py

Feature map dimensions (matching the spec):
  HSV histogram:   8 bins × 3 channels = 24
  Optical flow:    1
  Edge density:    1
  Entropy:         1
  Turbidity proxy: 1
  ─────────────────────────────────────
  d_raw = 28  →  projected to d_v = 128
"""

import math
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple

from utils import get_logger

logger = get_logger('V-Agent')

# Spec constants
D_RAW  = 28   # raw frame feature vector dimension (HSV 24 + flow + edge + entropy + turbidity)
D_V    = 128  # vision embedding dimension (d_v in spec)
T_SEQ  = 50   # temporal sequence length


# ─────────────────────────────────────────────────────────────────────────────
# Physics-based synthetic frame feature extractor  φ(F_t)
# ─────────────────────────────────────────────────────────────────────────────

def _simulate_frame_features(
    t: float,
    sa: float,
    mw: float,
    logp: float,
) -> np.ndarray:
    """
    Simulate a single frame's feature vector x_t ∈ R^{d_raw=28}.

    Channels (matching the spec):
      [0:24] HSV histogram (8 bins × 3 channels)
      [24]   Optical flow magnitude
      [25]   Edge density
      [26]   Entropy
      [27]   Turbidity proxy (intensity variance)

    Args:
        t:    Normalised time ∈ [0, 1].
        sa:   norm_sa:  1 - (SA - 1)/9  ∈ [0,1].  Higher = easier to synthesize.
        mw:   norm_mw:  exp(-MolWt/500) ∈ [0,1].
        logp: norm_lp:  gaussian(logP, 2.5) ∈ [0,1].
    """
    feats = np.zeros(D_RAW, dtype=np.float32)

    # SA drives reaction turbulence: harder molecules → noisier histograms
    turb = 1.0 - sa
    decay = math.exp(-2.0 * t)   # reactions start fast, settle

    # ── HSV histogram (24 dims) ──────────────────────────────────────────────
    for bin_idx in range(8):
        phase_h = bin_idx * math.pi / 8
        phase_s = bin_idx * math.pi / 6
        phase_v = bin_idx * math.pi / 4
        noise = np.random.normal(0, 0.05 + 0.3 * turb)
        feats[bin_idx]      = abs(mw  * math.sin(2 * math.pi * t + phase_h) * decay) + noise
        feats[bin_idx + 8]  = abs(sa  * math.cos(3 * math.pi * t + phase_s) * decay) + noise
        feats[bin_idx + 16] = abs(logp * math.sin(4 * math.pi * t + phase_v) * decay) + noise

    # ── Optical flow magnitude [24] ──────────────────────────────────────────
    # High early (reactants moving), decays as equilibrium is reached
    feats[24] = mw * decay + np.random.normal(0, 0.1 * turb)

    # ── Edge density [25] ────────────────────────────────────────────────────
    # Precipitate/crystal formation → sharper edges mid-reaction
    feats[25] = sa * abs(math.sin(3 * math.pi * t)) + np.random.normal(0, 0.05)

    # ── Entropy [26] ─────────────────────────────────────────────────────────
    # Disorder peaks mid-reaction, collapses at end
    feats[26] = turb * math.sin(math.pi * t) + np.random.normal(0, 0.05)

    # ── Turbidity proxy [27] — intensity variance ────────────────────────────
    feats[27] = turb * decay + np.random.normal(0, 0.1 * turb)

    return np.clip(feats, -3.0, 3.0)


def simulate_reaction_video(
    mol_feats: np.ndarray,
    seq_len: int = T_SEQ,
    feature_dim: int = D_RAW,
) -> torch.Tensor:
    """
    Simulate a reaction video as a time-series of frame features.

    For each frame t, computes x_t = φ(F_t) from the molecular descriptors,
    matching the spec's HSV/flow/edge/entropy/turbidity decomposition.

    Args:
        mol_feats:   (3,) array [norm_sa, norm_mw, norm_lp]. None returns zero.
        seq_len:     Temporal length T (default 50).
        feature_dim: d_raw (default 28). Passed for API consistency.

    Returns:
        Tensor of shape (T, d_raw).
    """
    if mol_feats is None or len(mol_feats) < 3:
        return torch.zeros((seq_len, D_RAW), dtype=torch.float)

    sa, mw, logp = float(mol_feats[0]), float(mol_feats[1]), float(mol_feats[2])
    frames = np.zeros((seq_len, D_RAW), dtype=np.float32)

    for i in range(seq_len):
        t_norm = i / max(1, seq_len - 1)   # normalised time ∈ [0, 1]
        frames[i] = _simulate_frame_features(t_norm, sa, mw, logp)

    return torch.tensor(frames, dtype=torch.float)


# ─────────────────────────────────────────────────────────────────────────────
# Positional Encoding  PE(t)
# ─────────────────────────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding — Section 2 of the spec."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, L, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D)  →  x + PE"""
        return x + self.pe[:, :x.size(1), :]


# ─────────────────────────────────────────────────────────────────────────────
# Attention Pooling  h_vision = Σ α_t E_t
# ─────────────────────────────────────────────────────────────────────────────

class AttentionPooling(nn.Module):
    """
    Spec Section 4: attention-weighted temporal pooling.

        α_t = softmax(w^T E_t)
        h   = Σ α_t · E_t
    """
    def __init__(self, embed_dim: int):
        super().__init__()
        self.w = nn.Linear(embed_dim, 1, bias=False)  # learnable w ∈ R^{d_v}

    def forward(self, E: torch.Tensor) -> torch.Tensor:
        """
        E: (B, T, D)
        Returns: (B, D)
        """
        scores = self.w(E)              # (B, T, 1)
        alpha  = torch.softmax(scores, dim=1)  # (B, T, 1)
        pooled = (alpha * E).sum(dim=1)         # (B, D)
        return pooled


# ─────────────────────────────────────────────────────────────────────────────
# VisionTemporalTransformer
# ─────────────────────────────────────────────────────────────────────────────

class VisionTemporalTransformer(nn.Module):
    """
    Full VisionTemporalTransformer following the MATMED research spec:

      1. Project x_t → e_t via W_e  (input_proj)
      2. Add positional encoding  ẽ_t = e_t + PE(t)
      3. L=3 Transformer encoder layers (multi-head self-attention + FFN + LayerNorm + residual)
      4. Attention pooling to get h_vision ∈ R^{d_v}
    """

    def __init__(
        self,
        input_dim: int = D_RAW,    # d_raw = 28
        embed_dim: int = D_V,      # d_v   = 128
        num_layers: int = 3,
        num_heads: int = 4,
        ff_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # W_e: projection x_t ∈ R^{d_raw} → e_t ∈ R^{d_v}
        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim)

        # L transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='relu',     # spec uses ReLU in FFN
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Attention pooling (replaces mean pooling in the spec)
        self.attn_pool = AttentionPooling(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_raw) raw video features.
        Returns:
            h_vision: (B, d_v) attention-pooled vision embedding.
        """
        e   = self.input_proj(x)         # (B, T, d_v)       W_e x_t + b_e
        e   = self.pos_encoder(e)        # (B, T, d_v)       ẽ_t = e_t + PE(t)
        E   = self.transformer(e)        # (B, T, d_v)       L transformer layers
        h   = self.attn_pool(E)          # (B, d_v)          attention pooling
        return h


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('Testing Research-Grade Vision Agent...\n')

    # Simulate easy vs hard molecules
    easy_feats = np.array([0.90, 0.75, 0.85], dtype=np.float32)  # Easy synthesis
    hard_feats = np.array([0.10, 0.20, 0.15], dtype=np.float32)  # Hard synthesis

    easy_vid = simulate_reaction_video(easy_feats)
    hard_vid = simulate_reaction_video(hard_feats)

    print(f'Video shape (T=50, d_raw=28): {easy_vid.shape}')
    print(f'Easy mol variance: {torch.var(easy_vid):.4f}')
    print(f'Hard mol variance: {torch.var(hard_vid):.4f}')
    assert easy_vid.shape == (50, 28), 'Wrong shape!'

    # Test transformer
    vtt = VisionTemporalTransformer(input_dim=D_RAW, embed_dim=D_V, num_layers=3, num_heads=4)
    print(f'\nVisionTemporalTransformer params: {sum(p.numel() for p in vtt.parameters()):,}')

    h = vtt(easy_vid.unsqueeze(0))
    print(f'Output h_vision shape: {h.shape}')
    assert h.shape == (1, D_V)
    print('\n✅ Research-grade Vision Agent self-test passed!')

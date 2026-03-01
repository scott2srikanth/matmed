"""
vision_agent.py — Phase 3: Vision-Agent Integration
=====================================================
Simulates a reaction video signal based on physicochemical descriptors,
and processes it through a VisionTemporalTransformer to produce a summarizing
embedding. This embedding is used by the Reaction Agent to better predict
synthesis yield.
"""

import math
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple

from utils import get_logger

logger = get_logger('V-Agent')


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic Reaction Video Simulator
# ─────────────────────────────────────────────────────────────────────────────

def simulate_reaction_video(feats: np.ndarray, seq_len: int = 50, feature_dim: int = 16) -> torch.Tensor:
    """
    Simulates a time-series 'video' tensor representing the chemical reaction.

    Args:
        feats: (3,) numpy array containing [norm_sa, norm_mw, norm_lp].
               These dictate the amplitude and frequency of the simulated signal.
        seq_len: Length of the time-series sequence (T).
        feature_dim: Dimension of each frame in the sequence.

    Returns:
        A torch.Tensor of shape (seq_len, feature_dim).
        Returns a zero-tensor if feats is None or invalid.
    """
    if feats is None or len(feats) != 3:
        return torch.zeros((seq_len, feature_dim), dtype=torch.float)

    # feats contains: [norm_sa, norm_mw, norm_lp]
    freq   = 1.0 + 5.0 * (1.0 - feats[0])  # Lower SA (harder synthesis) -> higher noise frequency
    amp    = 0.5 + 2.0 * feats[1]          # Lighter molecules -> higher amplitude
    decay  = 0.1 + feats[2]                # Lipophilicity -> controls signal decay over time

    t = np.linspace(0, 10, seq_len)
    signal = np.zeros((seq_len, feature_dim), dtype=np.float32)

    for i in range(feature_dim):
        # Generate a mixture of sinusoidal waves with unique per-feature phase shifts
        phase = i * math.pi / feature_dim
        base_wave = amp * np.sin(freq * t + phase) * np.exp(-decay * t / 10.0)
        
        # Add normally distributed noise representing chaotic reaction intermediates
        noise = np.random.normal(0, 0.1 + 0.5 * (1.0 - feats[0]), size=seq_len)
        signal[:, i] = base_wave + noise

    return torch.tensor(signal, dtype=torch.float)


# ─────────────────────────────────────────────────────────────────────────────
# Vision Temporal Transformer
# ─────────────────────────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encodings for the temporal sequence."""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x shape: (batch_size, seq_len, d_model)"""
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class VisionTemporalTransformer(nn.Module):
    """
    Processes the raw simulated reaction video sequence into a fixed-size embedding.
    """
    def __init__(
        self,
        input_dim: int = 16,
        embed_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        ff_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Project raw feature dimension -> transformer embedding dimension
        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Final pooling projection to summarize the sequence into one vector
        self.pooler = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, input_dim) tensor of vision features.
        Returns:
            (batch_size, embed_dim) summarizing embedding.
        """
        # (B, T, D)
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        
        # (B, T, D) -> (B, T, D)
        out_seq = self.transformer(x)
        
        # Temporal pooling (mean across seq_len)
        # (B, D)
        pooled = out_seq.mean(dim=1)
        return self.pooler(pooled)


if __name__ == '__main__':
    # Self-test validation
    print("Testing Vision Component...")
    
    # 1. Simulate a reaction video for a difficult molecule (SA=1.5, MW=400, logP=3.0)
    # Feats normalized logic: SA(1.5)->0.94, MW(400)->0.44, logP(3.0)->0.93
    dummy_feats = np.array([0.94, 0.44, 0.93], dtype=np.float32)
    video_tensor = simulate_reaction_video(dummy_feats, seq_len=50, feature_dim=16)
    
    print(f"Simulated Video Tensor Shape: {video_tensor.shape}")
    print(f"Signal Variance (differs by molecule): {torch.var(video_tensor):.4f}")
    
    # 2. Test the Transformer
    model = VisionTemporalTransformer(input_dim=16, embed_dim=128, num_layers=3, num_heads=4)
    print(f"Transformer Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Add batch dimension
    batched_video = video_tensor.unsqueeze(0)
    embedding = model(batched_video)
    
    print(f"Final Summarized Vision Embedding Shape: {embedding.shape}")
    assert embedding.shape == (1, 128)
    print("✅ Vision Agent self-test passed!")

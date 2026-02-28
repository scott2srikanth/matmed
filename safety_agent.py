"""
safety_agent.py — S-Agent: ADMET Critic
==========================================
Implements a Transformer encoder that predicts ADMET properties
(Absorption, Distribution, Metabolism, Excretion, Toxicity) for a given
SMILES string.

Scientific rationale:
  >90% of drug candidates fail in clinical trials due to ADMET issues, not
  lack of efficacy.  A dedicated ADMET critic agent allows the policy to
  penalise molecules likely to have poor pharmacokinetics *before* expensive
  wet-lab assays are required.

Implementation strategy:
  1. Attempt to load ChemBERTa (seyonec/ChemBERTa-zinc-base-v1) from
     HuggingFace as a pretrained molecular encoder.
  2. Fall back to a lightweight character-level Transformer encoder if
     ChemBERTa is unavailable or the network is off.

Output:
  - toxicity_prob : float ∈ [0, 1]  (higher → more toxic)
  - admet_score   : float ∈ [0, 1]  (higher → more drug-like)
  - embedding     : (hidden_dim,) tensor

Simplifications for prototype:
  - ADMET scores are regression outputs of the learned encoder + MLP head.
  - No actual ADMET dataset is used; scores are indicative proxy signals.
"""

import math
import logging
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import SMILESTokenizer, set_seed, get_logger

logger = get_logger('S-Agent')


# ─────────────────────────────────────────────────────────────────────────────
# Fallback: simple character-level Transformer encoder
# ─────────────────────────────────────────────────────────────────────────────

class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]          # type: ignore[index]
        return self.dropout(x)


class _SimpleTransformerEncoder(nn.Module):
    """
    Lightweight Transformer encoder on top of character-level SMILES tokens.
    Used as the fallback when ChemBERTa is unavailable.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.1,
        max_len: int = 128,
        pad_idx: int = 0,
    ) -> None:
        super().__init__()
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_enc   = _PositionalEncoding(d_model, max_len=max_len + 2, dropout=dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, activation='gelu',
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.ln = nn.LayerNorm(d_model)
        self.d_model = d_model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (batch, seq_len) token ids.

        Returns:
            (batch, d_model) mean-pooled encoding.
        """
        key_pad_mask = (input_ids == self.pad_idx)
        x = self.embedding(input_ids) * math.sqrt(self.d_model)
        x = self.pos_enc(x)
        h = self.encoder(x, src_key_padding_mask=key_pad_mask)  # (B, L, D)
        # Mean-pool over non-padding positions
        active = (~key_pad_mask).float().unsqueeze(-1)
        emb = (self.ln(h) * active).sum(1) / active.sum(1).clamp(min=1)
        return emb  # (B, d_model)


# ─────────────────────────────────────────────────────────────────────────────
# ADMET Head
# ─────────────────────────────────────────────────────────────────────────────

class _ADMETHead(nn.Module):
    """
    MLP head that takes a molecular embedding and outputs:
      - toxicity_prob  (Sigmoid) ∈ [0, 1]
      - admet_score    (Sigmoid) ∈ [0, 1]
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 2),   # [toxicity_logit, admet_logit]
        )

    def forward(self, emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            emb: (batch, d_model) or (d_model,)

        Returns:
            toxicity_prob: tensor ∈ [0, 1]
            admet_score:   tensor ∈ [0, 1]
        """
        out = self.net(emb)       # (..., 2)
        tox  = torch.sigmoid(out[..., 0])
        admet = torch.sigmoid(out[..., 1])
        return tox, admet


# ─────────────────────────────────────────────────────────────────────────────
# Safety Agent
# ─────────────────────────────────────────────────────────────────────────────

class SafetyAgent(nn.Module):
    """
    S-Agent: ADMET Critic for drug-likeness evaluation.

    Attempts to use a pretrained ChemBERTa encoder; falls back to a
    simple character-level Transformer if unavailable.

    Usage::

        agent = SafetyAgent()
        tox, admet, emb = agent("CC(=O)Oc1ccccc1C(=O)O")
    """

    CHEMBERTA_MODEL = "seyonec/ChemBERTa-zinc-base-v1"

    def __init__(
        self,
        d_model: int = 128,
        max_len: int = 128,
        use_chemberta: bool = True,
    ) -> None:
        super().__init__()
        self.max_len = max_len
        self.tokenizer_smiles = SMILESTokenizer()
        self._use_chemberta = False
        self._hf_tokenizer = None

        if use_chemberta:
            self._try_load_chemberta(d_model)

        if not self._use_chemberta:
            logger.info("Using fallback Transformer encoder for S-Agent.")
            self.encoder = _SimpleTransformerEncoder(
                vocab_size=self.tokenizer_smiles.vocab_size,
                d_model=d_model,
                max_len=max_len,
                pad_idx=self.tokenizer_smiles.pad_idx,
            )
            self.embed_dim = d_model
        
        self.admet_head = _ADMETHead(self.embed_dim)

    def _try_load_chemberta(self, d_model: int) -> None:
        """Attempt to load ChemBERTa; set flags on success."""
        try:
            from transformers import AutoTokenizer, AutoModel  # type: ignore
            logger.info(f"Loading ChemBERTa: {self.CHEMBERTA_MODEL}")
            self._hf_tokenizer = AutoTokenizer.from_pretrained(self.CHEMBERTA_MODEL)
            cb_model = AutoModel.from_pretrained(self.CHEMBERTA_MODEL)

            # Projection to uniform d_model
            cb_hidden = cb_model.config.hidden_size
            self.encoder = cb_model
            self.proj = nn.Linear(cb_hidden, d_model)
            self.embed_dim = d_model
            self._use_chemberta = True
            logger.info(f"ChemBERTa loaded (hidden_size={cb_hidden}).")
        except Exception as exc:
            logger.warning(f"ChemBERTa unavailable ({exc}). Using fallback encoder.")

    def _encode_chemberta(self, smiles: str) -> torch.Tensor:
        """Encode a SMILES with ChemBERTa and project to embed_dim."""
        device = next(self.parameters()).device
        inputs = self._hf_tokenizer(
            smiles,
            return_tensors='pt',
            max_length=self.max_len,
            truncation=True,
            padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self.encoder(**inputs)
        # CLS token
        cls_emb = out.last_hidden_state[:, 0, :]  # (1, hidden)
        return self.proj(cls_emb).squeeze(0)       # (embed_dim,)

    def _encode_fallback(self, smiles: str) -> torch.Tensor:
        """Encode a SMILES with the fallback character-level encoder."""
        device = next(self.parameters()).device
        ids = self.tokenizer_smiles.encode(smiles, max_len=self.max_len)
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)  # (1, L)
        return self.encoder(input_ids).squeeze(0)   # (embed_dim,)

    def forward(self, smiles: str) -> Tuple[float, float, torch.Tensor]:
        """
        Predict toxicity and ADMET score for a SMILES string.

        Args:
            smiles: SMILES string of the candidate molecule.

        Returns:
            toxicity_prob: float ∈ [0, 1].
            admet_score:   float ∈ [0, 1].
            embedding:     (embed_dim,) tensor — passed to P-Agent.
        """
        if self._use_chemberta:
            emb = self._encode_chemberta(smiles)
        else:
            emb = self._encode_fallback(smiles)

        tox, admet = self.admet_head(emb)
        return float(tox.item()), float(admet.item()), emb


# ─────────────────────────────────────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    from utils import get_zinc_sample
    set_seed(42)

    agent = SafetyAgent(use_chemberta=True)
    print(f"S-Agent embed_dim: {agent.embed_dim}")
    print(f"S-Agent parameters: {sum(p.numel() for p in agent.parameters()):,}")

    for smi in get_zinc_sample()[:5]:
        tox, admet, emb = agent(smi)
        print(f"  tox={tox:.4f}  admet={admet:.4f}  emb_shape={emb.shape}  smi={smi[:40]}")

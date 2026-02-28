"""
generator_agent.py — G-Agent: Molecular Generator
====================================================
Implements a **decoder-only Transformer** that generates novel molecular
structures as SMILES strings.

Scientific rationale:
  Analogous to language modelling, molecules can be written as character
  sequences in SMILES notation.  A causal (decoder-only) Transformer trained
  with teacher forcing learns the statistical distribution of valid, drug-like
  SMILES strings.  At inference time, autoregressive sampling produces novel
  candidates not present in the training dataset.

Architecture summary:
  - Token embedding (vocab_size → d_model=256)
  - Sinusoidal positional encoding
  - 4 x causally-masked Transformer decoder layers (4 heads, d_ff=512)
  - Output projection → logits over vocabulary
  - Latent embedding: mean-pooled hidden states of the final layer

References:
  - "Attention is All You Need" (Vaswani et al., 2017)
  - "Molecular Transformer" (Schwaller et al., 2019)
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import SMILESTokenizer, is_valid_smiles, set_seed, get_zinc_sample

# ─────────────────────────────────────────────────────────────────────────────
# Positional Encoding
# ─────────────────────────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    """
    Fixed sinusoidal positional encoding added to token embeddings.
    Gives the model information about token position in the sequence.
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input embeddings."""
        x = x + self.pe[:, :x.size(1), :]            # type: ignore[index]
        return self.dropout(x)


# ─────────────────────────────────────────────────────────────────────────────
# Generator Agent
# ─────────────────────────────────────────────────────────────────────────────

class GeneratorAgent(nn.Module):
    """
    G-Agent: Decoder-only Transformer for SMILES generation.

    Attributes:
        tokenizer:  Shared SMILESTokenizer.
        d_model:    Hidden dimension (default 256).
        nhead:      Number of attention heads (default 4).
        num_layers: Number of Transformer decoder layers (default 4).
        max_len:    Maximum sequence length (default 128).
    """

    def __init__(
        self,
        tokenizer: Optional[SMILESTokenizer] = None,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
        max_len: int = 128,
    ) -> None:
        super().__init__()

        self.tokenizer = tokenizer or SMILESTokenizer()
        vocab_size = self.tokenizer.vocab_size
        self.d_model = d_model
        self.max_len = max_len

        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=self.tokenizer.pad_idx)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len + 2, dropout=dropout)

        # Causal Transformer decoder stack (each layer is encoder-style but
        # with a causal mask so it behaves as a decoder)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection to vocabulary logits
        self.output_proj = nn.Linear(d_model, vocab_size, bias=True)

        # Layer norm applied before pooling to produce a stable embedding
        self.ln = nn.LayerNorm(d_model)

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier-uniform initialisation for projection layers."""
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    @staticmethod
    def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        """Generate an upper-triangular causal attention mask (True = masked)."""
        return torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()

    def forward(
        self, input_ids: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with teacher forcing.

        Args:
            input_ids:        (batch, seq_len) token ids — shifted-right target sequence.
            key_padding_mask: (batch, seq_len) bool tensor, True where padding.

        Returns:
            logits:    (batch, seq_len, vocab_size) un-normalised logit scores.
            embedding: (batch, d_model) mean-pooled latent representation.
        """
        seq_len = input_ids.size(1)
        device = input_ids.device

        x = self.embedding(input_ids) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)

        causal_mask = self._causal_mask(seq_len, device)

        hidden = self.transformer(
            x,
            mask=causal_mask,
            src_key_padding_mask=key_padding_mask,
        )  # (batch, seq_len, d_model)

        logits = self.output_proj(hidden)  # (batch, seq_len, vocab_size)

        # Mean pool over non-padding positions
        if key_padding_mask is not None:
            active = (~key_padding_mask).float().unsqueeze(-1)  # (B, L, 1)
            embedding = (self.ln(hidden) * active).sum(1) / active.sum(1).clamp(min=1)
        else:
            embedding = self.ln(hidden).mean(dim=1)  # (batch, d_model)

        return logits, embedding

    @torch.no_grad()
    def generate(
        self,
        batch_size: int = 1,
        temperature: float = 1.0,
        top_k: int = 0,
        device: Optional[torch.device] = None,
    ) -> Tuple[list, torch.Tensor]:
        """
        Autoregressively generate SMILES strings.

        Args:
            batch_size:  Number of molecules to generate simultaneously.
            temperature: Sampling temperature (higher → more diverse).
            top_k:       If > 0, restrict sampling to top-k tokens.
            device:      Torch device.

        Returns:
            smiles_list: List of decoded SMILES strings (length = batch_size).
            embeddings:  (batch_size, d_model) latent embeddings.
        """
        if device is None:
            device = next(self.parameters()).device

        self.eval()
        sos = self.tokenizer.sos_idx
        eos = self.tokenizer.eos_idx
        pad = self.tokenizer.pad_idx

        # Initialise sequences with SOS
        sequences = torch.full((batch_size, 1), sos, dtype=torch.long, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        embeddings: Optional[torch.Tensor] = None

        for _ in range(self.max_len - 1):
            logits, emb = self.forward(sequences)
            embeddings = emb  # update embedding each step

            # Sample next token from last position
            next_logits = logits[:, -1, :] / temperature  # (B, vocab)

            if top_k > 0:
                topk_vals = next_logits.topk(top_k, dim=-1).values[:, -1:]
                next_logits = next_logits.masked_fill(next_logits < topk_vals, -float('inf'))

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # Lock finished sequences to PAD
            next_token[finished] = pad

            sequences = torch.cat([sequences, next_token], dim=1)

            # Mark newly finished sequences
            finished = finished | (next_token.squeeze(-1) == eos)
            if finished.all():
                break

        smiles_list = [
            self.tokenizer.decode(sequences[i].tolist()) for i in range(batch_size)
        ]

        assert embeddings is not None
        return smiles_list, embeddings

    # ── Training helpers ────────────────────────────────────────────────────

    def compute_loss(
        self, src: torch.Tensor, tgt: torch.Tensor, pad_idx: Optional[int] = None
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss with teacher forcing.

        Args:
            src: (batch, seq_len) input token ids (e.g., sequence[:-1]).
            tgt: (batch, seq_len) target token ids (e.g., sequence[1:]).
            pad_idx: Index of the padding token (ignored in loss).

        Returns:
            Scalar loss tensor.
        """
        pad = pad_idx if pad_idx is not None else self.tokenizer.pad_idx
        key_pad_mask = (src == pad)

        logits, _ = self.forward(src, key_padding_mask=key_pad_mask)
        # logits: (B, L, V); tgt: (B, L)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            tgt.reshape(-1),
            ignore_index=pad,
        )
        return loss


# ─────────────────────────────────────────────────────────────────────────────
# Pretraining helper (used by train_matmed.py)
# ─────────────────────────────────────────────────────────────────────────────

def pretrain_generator(
    agent: GeneratorAgent,
    smiles_list: list,
    num_epochs: int = 5,
    lr: float = 1e-3,
    device: Optional[torch.device] = None,
) -> None:
    """
    Quick supervised pretraining of the generator on a SMILES corpus.

    Uses teacher-forcing: given the sequence [SOS, t1, t2, ...], predict
    [t1, t2, ..., EOS].

    Args:
        agent:       GeneratorAgent instance.
        smiles_list: List of training SMILES strings.
        num_epochs:  Number of full passes over the data.
        lr:          Learning rate for Adam.
        device:      Torch device (auto-detected if None).
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    agent = agent.to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=lr)

    tok = agent.tokenizer
    encoded = [tok.encode(s, max_len=agent.max_len) for s in smiles_list]

    agent.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for ids in encoded:
            seq = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
            src = seq[:, :-1]   # drop last token
            tgt = seq[:, 1:]    # drop SOS

            loss = agent.compute_loss(src, tgt)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        avg = total_loss / max(1, len(encoded))
        print(f"  [G-Agent pretrain] Epoch {epoch + 1}/{num_epochs}  loss={avg:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    set_seed(42)
    tok = SMILESTokenizer()
    agent = GeneratorAgent(tokenizer=tok)

    print(f"G-Agent parameters: {sum(p.numel() for p in agent.parameters()):,}")

    # Pretrain on tiny ZINC sample
    zinc = get_zinc_sample()
    pretrain_generator(agent, zinc, num_epochs=3)

    # Generate
    generated, embs = agent.generate(batch_size=4, temperature=0.8)
    print("\nGenerated SMILES:")
    for smi in generated:
        valid = "✓" if is_valid_smiles(smi) else "✗"
        print(f"  {valid}  {smi}")
    print(f"\nEmbedding shape: {embs.shape}")

"""
evaluator_agent.py — E-Agent: Binding Prediction (Graph Transformer)
======================================================================
Implements a **Graph Transformer** that predicts binding affinity from a
molecular graph derived from a SMILES string.

Scientific rationale:
  Molecular binding to a target protein is strongly determined by 3-D topology
  and electronic properties — both of which are encoded in the molecular graph.
  A Graph Transformer extends self-attention to graph-structured data, allowing
  each atom to attend to all other atoms while incorporating bond information
  as edge features.  Global mean-pooling aggregates atom representations into
  a fixed-size graph embedding used downstream by the P-Agent.

Architecture:
  SMILES → RDKit mol → atom/bond features → 3 Graph Transformer layers
  → global mean pool → MLP regression head → binding score ∈ [0, 1]

Node features per atom (7-dim):
  [atom_type_onehot (truncated), degree, hybridization_idx, is_aromatic]

Edge features per bond (4-dim):
  [bond_type_idx, is_conjugated, is_in_ring, stereo_idx]
"""

from typing import Tuple, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import rdchem

# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction from RDKit
# ─────────────────────────────────────────────────────────────────────────────

# Truncated atom-type vocabulary (most common in drug-like molecules)
ATOM_TYPES = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I', 'other']
ATOM_TYPE_IDX = {a: i for i, a in enumerate(ATOM_TYPES)}

HYBRIDIZATION_MAP = {
    rdchem.HybridizationType.SP:  0,
    rdchem.HybridizationType.SP2: 1,
    rdchem.HybridizationType.SP3: 2,
    rdchem.HybridizationType.SP3D: 3,
    rdchem.HybridizationType.SP3D2: 4,
}

BOND_TYPE_MAP = {
    rdchem.BondType.SINGLE:    0,
    rdchem.BondType.DOUBLE:    1,
    rdchem.BondType.TRIPLE:    2,
    rdchem.BondType.AROMATIC:  3,
}

NODE_FEAT_DIM = len(ATOM_TYPES) + 1 + 1 + 1  # one-hot + degree + hybrid + aromatic
EDGE_FEAT_DIM = 4 + 1 + 1 + 1                   # one-hot bond + conj + ring + stereo


def atom_features(atom: rdchem.Atom) -> List[float]:
    """Extract a numeric feature vector for one atom."""
    # One-hot atom type (length = len(ATOM_TYPES))
    sym = atom.GetSymbol()
    onehot = [0.0] * len(ATOM_TYPES)
    onehot[ATOM_TYPE_IDX.get(sym, ATOM_TYPE_IDX['other'])] = 1.0
    # Scalar features
    degree = atom.GetDegree() / 6.0          # normalise by max expected degree
    hyb = HYBRIDIZATION_MAP.get(atom.GetHybridization(), 5) / 5.0
    arom = float(atom.GetIsAromatic())
    return onehot + [degree, hyb, arom]


def bond_features(bond: rdchem.Bond) -> List[float]:
    """Extract a numeric feature vector for one bond (edge)."""
    # One-hot bond type
    onehot = [0.0] * 4
    onehot[BOND_TYPE_MAP.get(bond.GetBondType(), 0)] = 1.0
    conj  = float(bond.GetIsConjugated())
    ring  = float(bond.IsInRing())
    stereo = float(bond.GetStereo()) / 6.0   # normalise
    return onehot + [conj, ring, stereo]


def smiles_to_graph(smiles: str) -> Optional[dict]:
    """
    Convert a SMILES string to a simple graph dict compatible with this agent.

    Returns a dict with keys:
        'x':         (num_atoms, NODE_FEAT_DIM) float tensor — node features.
        'edge_index':(2, num_edges) long tensor  — COO-format edge indices.
        'edge_attr': (num_edges, EDGE_FEAT_DIM) float tensor — edge features.

    Returns None if the molecule cannot be parsed.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Node features
    node_feats = [atom_features(a) for a in mol.GetAtoms()]
    x = torch.tensor(node_feats, dtype=torch.float)  # (N, NODE_FEAT_DIM)

    # Edge indices and features (undirected → add both directions)
    edge_index_list: List[List[int]] = [[], []]
    edge_attr_list: List[List[float]] = []

    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        ef = bond_features(bond)
        for src, dst in [(i, j), (j, i)]:
            edge_index_list[0].append(src)
            edge_index_list[1].append(dst)
            edge_attr_list.append(ef)

    if not edge_index_list[0]:
        # Single-atom molecule — add a self-loop
        edge_index_list[0].append(0)
        edge_index_list[1].append(0)
        edge_attr_list.append([0.0] * EDGE_FEAT_DIM)

    edge_index = torch.tensor(edge_index_list, dtype=torch.long)
    edge_attr  = torch.tensor(edge_attr_list,  dtype=torch.float)

    return {'x': x, 'edge_index': edge_index, 'edge_attr': edge_attr}


# ─────────────────────────────────────────────────────────────────────────────
# Graph Transformer Layer (message-passing via attention)
# We implement a lightweight version without PyG dependency to keep
# the prototype self-contained while remaining structurally correct.
# ─────────────────────────────────────────────────────────────────────────────

class GraphTransformerLayer(nn.Module):
    """
    A single Graph Transformer layer.

    Each atom attends to ALL other atoms (dense attention) while
    edge features modulate the attention keys via an additive gate.
    This follows the formulation in "A Generalization of Transformers to
    Graphs" (Jain et al., 2021) simplified for clarity.
    """

    def __init__(self, d_model: int, nhead: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.d_model = d_model
        self.nhead   = nhead
        self.dk      = d_model // nhead

        self.q_proj  = nn.Linear(d_model, d_model)
        self.k_proj  = nn.Linear(d_model, d_model)
        self.v_proj  = nn.Linear(d_model, d_model)
        self.o_proj  = nn.Linear(d_model, d_model)

        # Edge-feature bias added to attention logits
        self.edge_bias = nn.Linear(EDGE_FEAT_DIM, nhead)

        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        h: torch.Tensor,          # (N, d_model) node features
        edge_index: torch.Tensor, # (2, E)
        edge_attr: torch.Tensor,  # (E, EDGE_FEAT_DIM)
    ) -> torch.Tensor:
        """
        Args:
            h:          Node feature matrix (N, d_model).
            edge_index: Edge indices in COO format (2, E).
            edge_attr:  Edge feature matrix (E, EDGE_FEAT_DIM).

        Returns:
            Updated node feature matrix (N, d_model).
        """
        N = h.size(0)
        res = h

        # Build dense adjacency & edge-feature bias matrices
        attn_bias = torch.zeros(N, N, self.nhead, device=h.device)
        if edge_index.size(1) > 0:
            src_idx, dst_idx = edge_index[0], edge_index[1]
            eb = self.edge_bias(edge_attr)             # (E, nhead)
            attn_bias[src_idx, dst_idx] = eb

        # Multi-head self-attention
        Q = self.q_proj(h).view(N, self.nhead, self.dk)  # (N, H, dk)
        K = self.k_proj(h).view(N, self.nhead, self.dk)
        V = self.v_proj(h).view(N, self.nhead, self.dk)

        # attn logits: (N, N, H)
        scores = torch.einsum('nhd,mhd->nmh', Q, K) / (self.dk ** 0.5) + attn_bias
        attn   = F.softmax(scores, dim=1)          # attend over neighbours dim

        # Aggregate values
        out = torch.einsum('nmh,mhd->nhd', attn, V).reshape(N, self.d_model)
        out = self.dropout(self.o_proj(out))

        h = self.ln1(res + out)
        h = h + self.dropout(self.ffn(h))
        return h


# ─────────────────────────────────────────────────────────────────────────────
# Evaluator Agent
# ─────────────────────────────────────────────────────────────────────────────

class EvaluatorAgent(nn.Module):
    """
    E-Agent: Graph Transformer for predicted binding affinity.

    Input:  SMILES string (converted internally to a graph).
    Output: binding_score ∈ [0, 1], graph_embedding (d_model,).

    The binding score is a *simulated proxy* (no real docking is performed).
    For real drug discovery, replace the MLP head with outputs from a docking
    engine (e.g., AutoDock Vina) or a structure-aware model.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        # Input projection: node features → hidden_dim
        self.node_proj = nn.Linear(NODE_FEAT_DIM, hidden_dim)

        # Graph Transformer layers
        self.gt_layers = nn.ModuleList([
            GraphTransformerLayer(d_model=hidden_dim, nhead=nhead, dropout=dropout)
            for _ in range(num_layers)
        ])

        # MLP regression head
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),           # output ∈ [0, 1]
        )

    def forward_graph(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch_idx: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process a single molecular graph.

        Args:
            x:          (N, NODE_FEAT_DIM) node feature tensor.
            edge_index: (2, E) edge indices.
            edge_attr:  (E, EDGE_FEAT_DIM) edge features.

        Returns:
            score:     Scalar binding score (1,).
            embedding: (hidden_dim,) global mean-pooled graph representation.
        """
        h = self.node_proj(x)  # (N, hidden_dim)

        for layer in self.gt_layers:
            h = layer(h, edge_index, edge_attr)

        if batch_idx is None:
            # Single graph: Global mean pooling
            embedding = h.mean(dim=0).unsqueeze(0)  # (1, hidden_dim)
        else:
            # Batched graphs: scatter mean pooling
            try:
                from torch_scatter import scatter_mean
                embedding = scatter_mean(h, batch_idx, dim=0)  # (batch_size, hidden_dim)
            except ImportError:
                # Manual scatter-mean fallback for pure PyTorch without PyG/torch_scatter
                num_graphs = int(batch_idx.max().item() + 1)
                embedding = torch.zeros(num_graphs, h.size(-1), device=h.device)
                counts = torch.zeros(num_graphs, 1, device=h.device)
                embedding.scatter_add_(0, batch_idx.unsqueeze(-1).expand_as(h), h)
                counts.scatter_add_(0, batch_idx.unsqueeze(-1), torch.ones_like(batch_idx.unsqueeze(-1).float()))
                embedding = embedding / counts.clamp(min=1)

        score = self.head(embedding).squeeze(-1)  # (batch_size,) or scalar if batch_size=1

        return score, embedding

    def forward(self, smiles: str) -> Tuple[float, torch.Tensor]:
        """
        End-to-end forward pass: SMILES → binding score + embedding.

        Args:
            smiles: SMILES string of the candidate molecule.

        Returns:
            binding_score: Float in [0, 1].
            embedding:     (hidden_dim,) tensor — passed to P-Agent.

        Raises:
            ValueError: If the SMILES cannot be parsed.
        """
        graph = smiles_to_graph(smiles)
        if graph is None:
            raise ValueError(f"Cannot parse SMILES: {smiles!r}")

        device = next(self.parameters()).device
        x          = graph['x'].to(device)
        edge_index = graph['edge_index'].to(device)
        edge_attr  = graph['edge_attr'].to(device)

        batch_idx = torch.zeros(x.size(0), dtype=torch.long, device=device)
        score, emb = self.forward_graph(x, edge_index, edge_attr, batch_idx)
        return float(score.item()), emb.squeeze(0)


# ─────────────────────────────────────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    from utils import get_zinc_sample, set_seed

    set_seed(42)
    agent = EvaluatorAgent()
    print(f"E-Agent parameters: {sum(p.numel() for p in agent.parameters()):,}")

    for smi in get_zinc_sample()[:5]:
        score, emb = agent.forward(smi)
        print(f"  binding={score:.4f}  emb_shape={emb.shape}  SMILES={smi[:40]}")

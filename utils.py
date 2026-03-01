"""
utils.py — Shared utilities for MATMED.

Provides:
  - Reproducible seed setting
  - SMILES tokenizer and vocabulary
  - SMILES validity checking via RDKit
  - Tanimoto diversity calculation
  - Molecule-to-graph conversion helpers
  - Logging / CSV metric saving
"""

import os
import random
import csv
import logging
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, DataStructs, AllChem
from rdkit import RDLogger

# Suppress RDKit C++ warnings/errors for invalid SMILES
RDLogger.DisableLog('rdApp.*')

# ─────────────────────────────────────────────────────────────
# Reproducibility & Data
# ─────────────────────────────────────────────────────────────

def get_zinc_sample(n: int = 10) -> List[str]:
    """
    Return a sample of ZINC SMILES strings.
    If n > 100, attempts to download a subset of `zpn/zinc250k` from HuggingFace.
    Fallback to a local dummy list if the download fails or n <= 100.
    """
    if n > 100:
        try:
            from datasets import load_dataset # type: ignore
            # Load the zinc dataset (often structured with a 'smiles' column)
            print(f"Downloading {n} ZINC SMILES from HuggingFace...")
            dataset = load_dataset("zpn/zinc250k", split="train")
            smiles_col = "smiles" if "smiles" in dataset.column_names else dataset.column_names[0]
            
            # Extract and shuffle slightly
            all_smiles = dataset[smiles_col]
            random.shuffle(all_smiles)
            return all_smiles[:n]
        except ImportError:
            print("Warning: `datasets` library not found. Run `pip install datasets` to download ZINC. Using dummy data instead.")
        except Exception as e:
            print(f"Failed to download ZINC: {e}. Using dummy data instead.")

    # A small set of valid SMILES strings (many from ZINC/ChEMBL)
    sample_smiles = [
        "CC(=O)Oc1ccccc1C(=O)O",          # Aspirin
        "CC1(C(N2C(S1)C(C2=O)NC(=O)CC3=CC=CC=C3)C(=O)O)C", # Penicillin G
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",    # Caffeine
        "CC(C)(C)NCC(O)c1ccc(O)c(CO)c1",   # Albuterol
        "c1ccccc1",                        # Benzene
        "CCO",                             # Ethanol
        "C1CCCCC1",                        # Cyclohexane
        "CC(=O)NC1=CC=C(O)C=C1",           # Paracetamol
        "C1=CC=C(C=C1)O",                  # Phenol
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"    # Ibuprofen
    ]
    result = []
    while len(result) < n:
        result.extend(sample_smiles)
    return result[:n]

def set_seed(seed: int = 42) -> None:
    """Set random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─────────────────────────────────────────────────────────────
# SMILES Tokenizer
# ─────────────────────────────────────────────────────────────

# Character-level vocabulary covering common SMILES tokens
SMILES_CHARS = (
    ['<PAD>', '<SOS>', '<EOS>', '<UNK>'] +
    list('BCNOPSFI') +                        # Organic subset atoms
    list('bcnops') +                           # Aromatic atoms
    list('0123456789') +
    list('()[]=#@+-\\./%') +
    ['Cl', 'Br', 'Si', 'Se', 'se']             # multi-char tokens
)

class SMILESTokenizer:
    """
    Character-level tokenizer for SMILES strings.
    Handles multi-character tokens (e.g., Cl, Br) before single characters.
    """

    PAD_TOKEN = '<PAD>'
    SOS_TOKEN = '<SOS>'
    EOS_TOKEN = '<EOS>'
    UNK_TOKEN = '<UNK>'

    def __init__(self, vocab: Optional[List[str]] = None):
        self.vocab = vocab if vocab is not None else SMILES_CHARS
        self.char2idx: Dict[str, int] = {c: i for i, c in enumerate(self.vocab)}
        self.idx2char: Dict[int, str] = {i: c for c, i in self.char2idx.items()}

        self.pad_idx = self.char2idx[self.PAD_TOKEN]
        self.sos_idx = self.char2idx[self.SOS_TOKEN]
        self.eos_idx = self.char2idx[self.EOS_TOKEN]
        self.unk_idx = self.char2idx[self.UNK_TOKEN]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    # Multi-char tokens sorted longest first so they match before single chars
    _MULTI = sorted(['Cl', 'Br', 'Si', 'Se', 'se'], key=len, reverse=True)

    def tokenize(self, smiles: str) -> List[str]:
        """Split a SMILES string into a list of token strings."""
        tokens: List[str] = []
        i = 0
        while i < len(smiles):
            matched = False
            for multi in self._MULTI:
                if smiles[i:i + len(multi)] == multi:
                    tokens.append(multi)
                    i += len(multi)
                    matched = True
                    break
            if not matched:
                tokens.append(smiles[i])
                i += 1
        return tokens

    def encode(self, smiles: str, max_len: int = 128,
               add_sos: bool = True, add_eos: bool = True) -> List[int]:
        """
        Encode a SMILES string to a list of integer indices.
        Pads / truncates to max_len (including special tokens).
        """
        tokens = self.tokenize(smiles)
        indices = [self.char2idx.get(t, self.unk_idx) for t in tokens]
        if add_sos:
            indices = [self.sos_idx] + indices
        if add_eos:
            indices = indices + [self.eos_idx]
        # Truncate
        indices = indices[:max_len]
        # Pad
        indices += [self.pad_idx] * (max_len - len(indices))
        return indices

    def decode(self, indices: List[int], strip_special: bool = True) -> str:
        """Decode a list of integer indices back to a SMILES string."""
        chars = []
        for idx in indices:
            tok = self.idx2char.get(idx, self.UNK_TOKEN)
            if strip_special and tok in (self.PAD_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN):
                continue
            chars.append(tok)
        return ''.join(chars)


# ─────────────────────────────────────────────────────────────
# SMILES / Molecular Validation
# ─────────────────────────────────────────────────────────────

def is_valid_smiles(smiles: str) -> bool:
    """Return True if `smiles` can be parsed by RDKit into a valid molecule."""
    if not smiles or not smiles.strip():
        return False
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None


def canonicalize(smiles: str) -> Optional[str]:
    """Return RDKit canonical SMILES or None if invalid."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def smiles_to_mol(smiles: str) -> Optional[Chem.Mol]:
    """Parse SMILES → RDKit Mol or return None."""
    return Chem.MolFromSmiles(smiles)


# ─────────────────────────────────────────────────────────────
# Diversity — Tanimoto Similarity
# ─────────────────────────────────────────────────────────────

def tanimoto_similarity(smiles1: str, smiles2: str) -> float:
    """Compute Morgan fingerprint Tanimoto similarity between two SMILES."""
    mol1, mol2 = Chem.MolFromSmiles(smiles1), Chem.MolFromSmiles(smiles2)
    if mol1 is None or mol2 is None:
        return 0.0
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, radius=2, nBits=1024)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius=2, nBits=1024)
    return DataStructs.TanimotoSimilarity(fp1, fp2)


def diversity_score(smiles_list: List[str]) -> float:
    """
    Mean pairwise Tanimoto DISTANCE (1 - similarity) across a list of SMILES.
    Higher values → more chemically diverse set.
    """
    valid = [s for s in smiles_list if is_valid_smiles(s)]
    if len(valid) < 2:
        return 0.0
    total, count = 0.0, 0
    for i in range(len(valid)):
        for j in range(i + 1, len(valid)):
            total += 1.0 - tanimoto_similarity(valid[i], valid[j])
            count += 1
    return total / count if count > 0 else 0.0


# ─────────────────────────────────────────────────────────────
# Molecular Descriptors (used by R-Agent)
# ─────────────────────────────────────────────────────────────

def compute_descriptors(smiles: str) -> Optional[Dict[str, float]]:
    """
    Compute a small set of physicochemical descriptors for a SMILES string.
    Returns a dict with 'mol_weight', 'logP', 'sa_score', or None if invalid.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        from rdkit.Chem import rdMolDescriptors as rdmd
        # Synthetic Accessibility score (lower = easier to synthesize)
        from rdkit.Contrib.SA_Score import sascorer  # type: ignore
        sa = sascorer.calculateScore(mol)
    except Exception:
        # SA_Score not always packaged; use a rough proxy
        sa = 5.0  # neutral default

    return {
        'mol_weight': Descriptors.MolWt(mol),
        'logP': Descriptors.MolLogP(mol),
        'sa_score': sa,
    }


# ─────────────────────────────────────────────────────────────
# Logging / Metrics
# ─────────────────────────────────────────────────────────────

def get_logger(name: str = 'matmed', level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s | %(name)s — %(message)s',
            datefmt='%H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def save_metrics_csv(metrics: List[Dict], filepath: str = 'matmed_metrics.csv') -> None:
    """Append a list of metric dicts to a CSV file."""
    if not metrics:
        return
    fieldnames = list(metrics[0].keys())
    write_header = not os.path.exists(filepath)
    with open(filepath, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(metrics)

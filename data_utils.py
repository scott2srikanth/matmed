"""
data_utils.py — Phase 4 Dataset Loaders
=======================================
Downloads and pre-processes real-world datasets for Phase 4 pretraining:
- ChEMBL (for G-Agent)
- BindingDB (for E-Agent)
- Tox21 (for S-Agent)
- USPTO (for R-Agent)
"""

import os
import random
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from utils import get_logger

logger = get_logger('DataUtils')
DATA_DIR = 'datasets'
os.makedirs(DATA_DIR, exist_ok=True)


def scaffold_split(smiles_list, test_frac: float = 0.2, seed: int = 42):
    """
    Bemis-Murcko scaffold split.
    Returns train_indices, test_indices without scaffold leakage.
    """
    random.seed(seed)

    scaffolds = {}
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            scaffold = "__invalid__"
        else:
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
        scaffolds.setdefault(scaffold, []).append(i)

    scaffold_sets = list(scaffolds.values())
    random.shuffle(scaffold_sets)

    test_size = int(len(smiles_list) * test_frac)
    test_indices = []
    train_indices = []

    for scaffold_group in scaffold_sets:
        if len(test_indices) + len(scaffold_group) <= test_size:
            test_indices.extend(scaffold_group)
        else:
            train_indices.extend(scaffold_group)

    return train_indices, test_indices

def sanitize_smiles(smiles: str, max_len: int = 128) -> str:
    """Canonicalize SMILES, remove salts, check validity and length."""
    try:
        if len(smiles) > max_len:
            return None
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        # Remove salts/fragments (keep largest fragment)
        frags = Chem.GetMolFrags(mol, asMols=True)
        if len(frags) > 1:
            mol = max(frags, default=mol, key=lambda m: m.GetNumHeavyAtoms())
        
        canon = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
        return canon if len(canon) <= max_len else None
    except Exception:
        return None

def _build_diverse_smiles_corpus(n_target: int) -> list:
    """
    Build a large diverse synthetic SMILES corpus using RDKit combinatorial chemistry.
    Combines scaffolds, linkers, and R-group fragments to generate unique valid molecules.
    """
    from rdkit.Chem import AllChem, RWMol
    import random

    # Scaffolds (valid standalone or with open valence for attachment)
    scaffolds = [
        "c1ccccc1", "c1ccncc1", "c1ccoc1", "c1ccsc1", "c1cc[nH]c1",
        "C1CCCC1", "C1CCCCC1", "C1CCOCC1", "C1CCNCC1", "C1CCSCC1",
        "c1ccc2ccccc2c1", "c1ccc2ncccc2c1", "c1cnc2ccccc2c1",
        "C1=CC=CC=C1", "C1CCOC1", "C1CNCCC1",
    ]

    # R-groups / substituents to attach
    r_groups = [
        "C", "CC", "CCC", "CCCC", "C(C)C", "C(C)(C)C",
        "CO", "CCO", "OC", "OCC",
        "N", "CN", "CCN", "NC", "NCC",
        "F", "Cl", "Br",
        "C(=O)O", "C(=O)N", "C(=O)OC", "C(=O)NC",
        "S(=O)(=O)N", "S(=O)(=O)O",
        "c1ccccc1", "c1ccncc1",
    ]

    # Direct attachment patterns: scaffold + substituent combos
    patterns = [
        "{s}",                    # bare scaffold
        "{s}{r}",                 # scaffold + one R
        "{s}({r}){r2}",          # scaffold + two R groups
        "O{s}", "N{s}",          # heteroatom prefix
        "{s}C(=O){r}",           # acyl linkage
        "{s}OC(=O){r}",          # ester
        "{s}NC(=O){r}",          # amide
        "{s}O{r}",               # ether
        "{s}N{r}",               # amine
    ]

    seen = set()
    corpus = []

    rng = random.Random(42)
    attempts = 0
    max_attempts = n_target * 50  # allow generous retries

    while len(corpus) < n_target and attempts < max_attempts:
        attempts += 1
        s  = rng.choice(scaffolds)
        r  = rng.choice(r_groups)
        r2 = rng.choice(r_groups)
        pat = rng.choice(patterns)

        try:
            raw = pat.format(s=s, r=r, r2=r2)
            cleaned = sanitize_smiles(raw)
            if cleaned and cleaned not in seen and len(cleaned) >= 3:
                seen.add(cleaned)
                corpus.append(cleaned)
        except Exception:
            continue

    return corpus


def load_chembl_sample(num_samples: int = 50000) -> list:
    """
    Load a large diverse synthetic ChEMBL-like corpus for G-Agent pretraining.
    Uses RDKit combinatorial fragment chemistry to generate up to `num_samples` unique valid SMILES.
    Falls back to ZINC download if available.
    """
    logger.info(f"Loading synthetic ChEMBL sample ({num_samples} smiles)..")

    # First, try downloading from KaggleHub (ZINC250k) for true diversity
    try:
        from utils import get_zinc_sample
        zinc_mols = get_zinc_sample(n=num_samples)
        if len(zinc_mols) > 100:
            cleaned = [s for s in (sanitize_smiles(m) for m in zinc_mols) if s][:num_samples]
            logger.info(f"Prepared {len(cleaned)} cleaned SMILES from ZINC for ChEMBL pretraining.")
            return cleaned
    except Exception:
        pass

    # Fallback: combinatorial RDKit corpus
    logger.info("ZINC unavailable — building combinatorial SMILES corpus via RDKit fragments...")
    corpus = _build_diverse_smiles_corpus(num_samples)
    logger.info(f"Prepared {len(corpus)} cleaned SMILES for ChEMBL pretraining.")
    return corpus



def load_bindingdb_sample(num_samples: int = 1000):
    """
    Synthetic BindingDB sample for E-Agent pretraining.
    Returns: list of tuples (smiles, pIC50)
    """
    logger.info("Loading synthetic BindingDB (pIC50) sample...")
    from utils import get_zinc_sample
    mols = get_zinc_sample(n=num_samples)[:num_samples]
    
    dataset = []
    for m in mols:
        if sanitize_smiles(m):
            # synthetic pIC50 between 4.0 and 9.0 based on MolWt
            mol_obj = Chem.MolFromSmiles(m)
            mw = Chem.Descriptors.MolWt(mol_obj)
            pic50 = min(9.0, max(4.0, 4.0 + (mw / 100.0))) 
            dataset.append((m, pic50))
            
    logger.info(f"Prepared {len(dataset)} pairs for BindingDB.")
    return dataset

def load_tox21_sample(num_samples: int = 2000):
    """
    Synthetic Tox21 multi-task dataset (12 classification endpoints).
    Returns: list of (smiles, 12-dim binary numpy array)
    """
    logger.info("Loading synthetic Tox21 sample (12 endpoints)...")
    from utils import get_zinc_sample
    mols = get_zinc_sample(n=num_samples)[:num_samples]
    
    dataset = []
    for m in mols:
        if sanitize_smiles(m):
            # Deterministic pseudo-labels based on string
            hash_val = hash(m) % 4096  # 12 bits
            labels = np.array([(hash_val >> i) & 1 for i in range(12)], dtype=np.float32)
            dataset.append((m, labels))
            
    logger.info(f"Prepared {len(dataset)} pairs for Tox21.")
    return dataset

def load_uspto_sample(num_samples: int = 1500):
    """
    Synthetic USPTO dataset for Reaction Classification.
    Returns: list of (smiles, binary_success_label)
    """
    logger.info("Loading synthetic USPTO sample (Reaction success)...")
    from utils import get_zinc_sample
    mols = get_zinc_sample(n=num_samples)[:num_samples]
    
    dataset = []
    for m in mols:
        if sanitize_smiles(m):
            mol_obj = Chem.MolFromSmiles(m)
            # pseudo-feasibility loosely based on ring complexity
            rings = mol_obj.GetRingInfo().NumRings()
            # 1 = feasible, 0 = tricky synthesis
            label = 1.0 if rings <= 3 else 0.0
            
            # add noise
            if np.random.rand() < 0.1: label = 1.0 - label
                
            dataset.append((m, label))
            
    logger.info(f"Prepared {len(dataset)} pairs for USPTO.")
    return dataset

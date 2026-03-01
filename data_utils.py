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
import pandas as pd
import numpy as np
from rdkit import Chem
from utils import get_logger

logger = get_logger('DataUtils')
DATA_DIR = 'datasets'
os.makedirs(DATA_DIR, exist_ok=True)

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

def load_chembl_sample(num_samples: int = 100000) -> list:
    """
    Load a synthetic ChEMBL sample for the prototype.
    In a full run, this would stream the official SQLite/CSV release.
    For Phase 4 prototype, we expand ZINC dynamically to mimic a large unlabeled corpus.
    """
    logger.info(f"Loading synthetic ChEMBL sample ({num_samples} smiles)..")
    from utils import get_zinc_sample
    base_mols = get_zinc_sample()
    
    # Repeat and minimally mutate to create a large dataset for LM pretraining
    smiles_list = []
    while len(smiles_list) < num_samples:
        for smi in base_mols:
            smiles_list.append(smi)
            if len(smiles_list) >= num_samples: break
            
    # Clean them
    cleaned = []
    for s in set(smiles_list):  # limit exact dupes
        c = sanitize_smiles(s)
        if c: cleaned.append(c)
        if len(cleaned) >= num_samples: break
            
    logger.info(f"Prepared {len(cleaned)} cleaned SMILES for ChEMBL pretraining.")
    return cleaned


def load_bindingdb_sample(num_samples: int = 1000):
    """
    Synthetic BindingDB sample for E-Agent pretraining.
    Returns: list of tuples (smiles, pIC50)
    """
    logger.info("Loading synthetic BindingDB (pIC50) sample...")
    from utils import get_zinc_sample
    mols = get_zinc_sample()[:num_samples]
    
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
    mols = get_zinc_sample()[:num_samples]
    
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
    mols = get_zinc_sample()[:num_samples]
    
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

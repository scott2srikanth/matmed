"""Explicit corpus provenance and scaffold-disjoint data; no synthetic fallback."""
import csv
import hashlib
import json
import random
from pathlib import Path

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from utils import SMILESTokenizer, SMILES_CHARS


def load_corpus(path, source, max_len=128):
    if not source.strip():
        raise ValueError("A dataset source/version or publication identifier is required")
    path = Path(path)
    with path.open() as f:
        reader = csv.DictReader(f)
        if 'smiles' not in (reader.fieldnames or []):
            raise ValueError("Corpus CSV requires a smiles column")
        raw = [row['smiles'] for row in reader]
    canonical = set()
    rejected = 0
    for smi in raw:
        mol = Chem.MolFromSmiles(smi)
        if mol is None or mol.GetNumAtoms() == 0:
            rejected += 1
        else:
            canonical.add(Chem.MolToSmiles(mol))
    # Expand coverage explicitly and save vocabulary WITH every checkpoint.
    base = SMILESTokenizer()
    extra = sorted({t for s in canonical for t in base.tokenize(s)} - set(SMILES_CHARS))
    tok = SMILESTokenizer(SMILES_CHARS + extra)
    kept = sorted(s for s in canonical if len(tok.tokenize(s)) + 2 <= max_len)
    metadata = dict(source=source, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    input_rows=len(raw), invalid_rows=rejected,
                    unique_valid=len(canonical), retained=len(kept),
                    too_long=len(canonical) - len(kept))
    if len(kept) < 3:
        raise ValueError("Corpus needs at least three distinct valid molecules")
    return kept, tok, metadata


def split_corpus(smiles, seed=42):
    groups = {}
    for smi in smiles:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(smiles=smi)
        groups.setdefault(scaffold, []).append(smi)
    if len(groups) < 3:
        raise ValueError("Need >=3 scaffolds, including the shared acyclic group; no random fallback")
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    # Split whole groups. Report actual molecule fractions, not nominal ratios.
    n_valid = max(1, int(len(keys) * .1))
    n_test = max(1, int(len(keys) * .1))
    subsets = [keys[n_valid + n_test:], keys[:n_valid], keys[n_valid:n_valid + n_test]]
    return {name: [s for key in subset for s in groups[key]]
            for name, subset in zip(('train', 'validation', 'test'), subsets)}


def write_manifest(path, metadata, split):
    Path(path).write_text(json.dumps({**metadata, 'split_seed': 42,
        'split_counts': {k: len(v) for k, v in split.items()},
        'splits': split}, indent=2))

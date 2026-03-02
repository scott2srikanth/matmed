from collections import defaultdict
import random

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def generate_scaffold(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold)


def scaffold_split(smiles_list, frac_train: float = 0.8, frac_valid: float = 0.1, seed: int = 42):
    random.seed(seed)
    scaffold_to_indices = defaultdict(list)

    for i, smi in enumerate(smiles_list):
        scaffold = generate_scaffold(smi)
        scaffold_to_indices[scaffold].append(i)

    scaffolds = list(scaffold_to_indices.keys())
    random.shuffle(scaffolds)

    train_cutoff = int(frac_train * len(scaffolds))
    valid_cutoff = int((frac_train + frac_valid) * len(scaffolds))

    train_scaffolds = scaffolds[:train_cutoff]
    valid_scaffolds = scaffolds[train_cutoff:valid_cutoff]
    test_scaffolds = scaffolds[valid_cutoff:]

    def indices_from_scaffolds(scaffold_subset):
        idx = []
        for s in scaffold_subset:
            idx.extend(scaffold_to_indices[s])
        return idx

    return (
        indices_from_scaffolds(train_scaffolds),
        indices_from_scaffolds(valid_scaffolds),
        indices_from_scaffolds(test_scaffolds),
    )

import os
import json
import numpy as np

from torch_geometric.datasets import QM9
from torch_geometric.data import Data

from scripts.metrics.molecule_conversion import QM9_to_ase, QM9_symbols
from scripts.metrics.ChemMetrics import ChemMetrics

# Disable valence logs
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

def basic_transform(data: Data):
    data.h = data.x[:, :5] # only keep the one-hot atom types 
    data.x = data.pos - data.pos.mean(dim=0) # zero mean
    del data.pos
    return data

def create_subset(dataset,
                  seed: int=42,
                  subset_size: int = 10000):
    np.random.seed(seed)
    num_total = len(dataset)

    # Get size and atom type distribution
    mol_by_size = {}
    for i, mol in enumerate(dataset):
        size = mol.h.shape[0]
        if size not in mol_by_size.keys():
            mol_by_size[size] = []
        mol_by_size[size].append(i)

    # Sample molecules proportionally
    selected_indices = []    
    for size in mol_by_size.keys():
        num_samples = int((len(mol_by_size[size]) / num_total) * subset_size)
        # Randomly molecules with this size
        sampled = np.random.choice(a=mol_by_size[size],
                                   size=num_samples, 
                                   replace=False)
        selected_indices.extend(sampled.tolist())
    
    # If short on target, add random mol from not already selected
    if len(selected_indices) < subset_size:
        num_samples = subset_size - len(selected_indices)
        excluded = list(set(range(num_total)) - set(selected_indices))
        sampled = np.random.choice(excluded, size=num_samples, replace=False)
        selected_indices.extend(sampled.tolist())
    # If over target, randomly exclude some of the selected
    if len(selected_indices) > subset_size:
        selected_indices = np.random.choice(selected_indices, 
                                           size=num_samples, 
                                           replace=False).tolist()
    return selected_indices


def process_QM9(seed: int=42,
                save_dir: str="data/QM9"):
    '''
    Downloads and process the QM9 module to use the nomenclature 
    from Hoogeboom (2022) - Equivariant Diffusion for Molecule Generation in 3D
    Divides the dataset into train, val, test splits and saves their metrics
    '''
    dataset = QM9(root=save_dir, pre_transform=basic_transform)
    # Split dataset into train-val-test
    num_total = len(dataset)
    num_train = 100000
    num_test = int(0.1 * num_total)
    num_val = num_total - (num_train + num_test)
    
    np.random.seed(seed)
    data_perm = np.random.permutation(x=num_total)
    train_split, val_split, test_split, extra_split = np.split(data_perm,
                                                  [num_train, num_train + num_val, num_train + num_val + num_test])
    assert len(extra_split) == 0, \
        f"Error during splitting {len(train_split)=}, {len(val_split)=}, {len(test_split)=}"
    
    train_subset = create_subset(dataset=dataset[train_split], seed=seed)
    splits = {"train": sorted(train_split.tolist()),
              "val": sorted(val_split.tolist()),
              "test": sorted(test_split.tolist()),
              "train_subset": sorted(train_subset)}
    
    splits_path = os.path.join(save_dir, "splits.json")
    with open(splits_path, "w") as f:
        json.dump(splits, f)

    print(f"Dataset splits saved to: {splits_path}")

    for split in splits:
        print(f"Processing {split}_split")
        idx = splits[split]
        molecules = dataset[idx]
        atoms = [QM9_to_ase(mol) for mol in molecules]
        split_metrics = ChemMetrics(atoms=atoms)
        print("Computing metrics")
        split_metrics.compute_metrics()
        summary = split_metrics.summarize()
    
        summary_path = os.path.join(save_dir, f"{split}_infos.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f)
        print(f"Summary of {split}_split save at {summary_path}")

        smiles_path = os.path.join(save_dir, f"{split}_smiles.json")
        with open(smiles_path, "w") as f:
            json.dump(split_metrics.smiles, f)
        print(f"SMILES of {split}_split saved at {smiles_path}")

def load_QM9_split(split: str,
                   data_dir: str="data/QM9",
                   load_info=False,
                   load_smiles=False,
                   as_ase_atoms=False):
    '''
    Load a QM9 split created by `process_QM9` 
    and optionally the relative summary
    '''
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {os.path.abspath(data_dir)}")
    dataset = QM9(root=data_dir, pre_transform=basic_transform)

    splits_file = os.path.join(data_dir, "splits.json")
    with open(splits_file, "r") as f:
        splits = json.load(f)
    split_idx = splits[split]
    split_set = dataset[split_idx]

    if as_ase_atoms:
        split_set = [QM9_to_ase(mol) for mol in split_set]

    split_infos = None
    if load_info:
        info_file = os.path.join(data_dir, f"{split}_infos.json")
        with open(info_file, "r") as f:
            split_infos = json.load(f)

    split_smiles = None
    if load_smiles:
        smile_file = os.path.join(data_dir, f"{split}_smiles.json")
        with open(smile_file, "r") as f:
            split_smiles = json.load(f)

    return split_set, split_infos, split_smiles

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='Seed to split dataset into train, val, test (default: %(default)s)')
    parser.add_argument('--save_dir', type=str, default='data/QM9', help='Directory where to save the dataset and infos (default: %(default)s)')
    args = parser.parse_args()
    
    process_QM9(seed=args.seed,
                save_dir=args.save_dir)
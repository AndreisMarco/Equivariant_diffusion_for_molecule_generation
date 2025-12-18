import numpy as np
from collections import Counter
from tqdm import tqdm

from ase import Atoms
from rdkit import Chem

from scripts.metrics.bonds import ALLOWED_BONDS
from scripts.metrics.molecule_conversion import ase_to_rdkit

class DiscreteHistogram():
    def __init__(self, bins: list[str]):
        self.histogram = {bin:0 for bin in bins}

    def update(self, values: list[str]):
        counter = Counter(values)
        for key in counter:
            if key not in self.histogram.keys():
                print(f"Trying to update not registered bin: {key} not in {list(self.histogram.keys())}")
                continue
            self.histogram[key] += counter[key]

    def get_histogram(self, normalize=False):
        if normalize:
            values = np.array(list(self.histogram.values()))
            normalized = {k: v/values.sum() for k, v in self.histogram.items()}
            return normalized
        else:
            return self.histogram
        
def check_stability(atoms: Atoms,
                    bond_order_per_atom: np.ndarray):
    num_atoms = len(atoms)
    '''
    Verifies if valence is respected for all atoms
    in the molecule
    '''
    stable_atoms = []
    atom_stable = 0
    for symbol_i, nr_bonds_i in zip(atoms.symbols, bond_order_per_atom):
        possible_bonds = ALLOWED_BONDS[symbol_i]
        if type(possible_bonds) == int:
            is_stable = (possible_bonds == nr_bonds_i)
        else:
            is_stable = nr_bonds_i in possible_bonds
        atom_stable += int(is_stable)
        if is_stable:
            stable_atoms.append(symbol_i)
    molecule_stable = (atom_stable == num_atoms)
    return molecule_stable, atom_stable, num_atoms, stable_atoms

def mol_to_smi(mol_: Chem.Mol):
        '''
        Tries to sanitize the molecule and returns it
        and its smile if is successful
        '''
        try:
            Chem.SanitizeMol(mol_)
            smi = Chem.MolToSmiles(mol_, canonical=True)
            return mol_, smi
        except ValueError:
            return None, None

def check_validity(mol: Chem.Mol):
    '''
    If mol_to_smi is successful, return the number 
    of disconnected fragments of the molecule, the largest fragment,
    its smile and whether it is valid
    '''
    mol, smi = mol_to_smi(mol)
    v, c = 0, 0
    if smi is not None:
        mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True)
        c = int(len(mol_frags) == 1)
        largest_mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
        mol, smi = mol_to_smi(largest_mol)
        v = int(smi is not None)
    return mol, smi, v, c

class ChemMetrics():
    '''
    Implements computation of the following metrics
    - valid
    - valid + connected
    - atom stability
    - molecule stability
    Also provides molecule size and atom type distribution
    '''
    def __init__(self, atoms: list[Atoms]):
        self.atoms = atoms 
        self.atom_types = self.find_atom_types(atoms)
        self.reset()
    
    def find_atom_types(self, atoms):
        seen = set()
        unique_atom_types = []
        for mol in atoms:
            for atom in mol:
                if atom.symbol not in seen:
                    seen.add(atom.symbol)
                    unique_atom_types.append(atom.symbol)
        return unique_atom_types
    
    def reset(self):
        self.smiles = []
        self.valid = []
        self.valid_connected = []
        self.atom_stable = []
        self.molecule_stable = []
        self.n_atoms = []
        self.atom_type_hist = DiscreteHistogram(bins=self.atom_types)
        self.atom_stability_hist = DiscreteHistogram(bins=self.atom_types)
    def compute_metrics(self):
        for a in tqdm(self.atoms, desc="Processing molecules"):
            raw_mol, bond_order_per_atom = ase_to_rdkit(a)
            mol, smi, v, c = check_validity(raw_mol)
            molecule_stable, atom_stable, n_atoms, stable_atoms = check_stability(a, bond_order_per_atom)
            if n_atoms == 4:
                print("Stop")
            self.smiles.append(smi)
            self.valid.append(v)
            self.valid_connected.append(c)
            self.atom_stable.append(atom_stable)
            self.molecule_stable.append(molecule_stable)
            self.n_atoms.append(n_atoms)
            self.atom_type_hist.update(a.symbols)
            self.atom_stability_hist.update(stable_atoms)
        self.mol_size_hist = Counter(self.n_atoms)

    def summarize(self):
        assert len(self.valid) == len(self.valid_connected)
        assert len(self.valid) == len(self.molecule_stable)
        assert len(self.valid) == len(self.smiles)

        n_samples = len(self.valid)
        n_atoms = sum(self.n_atoms)

        summary = {}
        summary["atom_stable"] = sum(self.atom_stable) / n_atoms
        summary["molecule_stable"] = sum(self.molecule_stable) / n_samples
        summary["valid"] = sum(self.valid) / n_samples
        summary["valid_connected"] = sum(self.valid_connected) / n_samples
        valid_unique_smiles = set([smiles for (v, smiles) in zip(self.valid, self.smiles) if v])
        summary["valid_unique"] = len(valid_unique_smiles) / n_samples
        summary["atom_type_hist"] = self.atom_type_hist.get_histogram()
        summary["atom_type_dist"] = self.atom_type_hist.get_histogram(normalize=True)
        summary["atom_stability_hist"] = self.atom_stability_hist.get_histogram(normalize=False)
        summary["mol_size_hist"] = self.mol_size_hist
        return summary


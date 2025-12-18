import numpy as np

from ase import Atoms
from rdkit import Chem
from torch_geometric.data import Data

from scripts.metrics.bonds import get_bond_order, BOND_LIST

QM9_symbols = ["H", "C", "N", "O", "F"] 

def QM9_to_ase(data: Data):
    '''
    Converts a molecule in torch_geometric.data.Data object 
    to a ase.Atoms object 
    '''
    atom_idxs = data.h.argmax(dim=1).tolist()  
    pos = data.x.numpy()
    atoms = Atoms(symbols=[QM9_symbols[i] for i in atom_idxs], positions=pos)
    return atoms

def ase_to_rdkit(atoms: Atoms,
                 single_bond: bool=False):
    '''
    Converts a molecule in the ase.Atoms object to 
    an Chem.RWMol object from RDKit 
    '''
    distances = atoms.get_all_distances()
    num_atoms = len(atoms)
    bond_order_per_atom = np.zeros(num_atoms, dtype="int")
    # Create molecule instance
    mol = Chem.RWMol()
    # Add all atoms first
    for symbol in atoms.symbols:
        a = Chem.Atom(symbol)
        mol.AddAtom(a)
    # Add all bonds
    for i in range(num_atoms):
        symbol_i = atoms.symbols[i]
        for j in range(i + 1, num_atoms):
            dist = distances[i, j]
            symbol_j = atoms.symbols[j]
            order = get_bond_order(
                symbol_i, symbol_j, dist, single_bond=single_bond
            )
            bond_order_per_atom[i] += order
            bond_order_per_atom[j] += order
            if order > 0:
                mol.AddBond(i, j, BOND_LIST[order])
    return mol, bond_order_per_atom
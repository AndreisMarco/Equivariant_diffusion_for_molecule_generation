import os
from collections import Counter
from ase.io import read
from scripts.metrics.ChemMetrics import ChemMetrics
from scripts.process_QM9 import load_QM9_split

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--xyz_file', type=str, required=True, help='File in xyz format containing the samples')
    parser.add_argument('--QM9_dir', type=str, default="data/QM9", help='Directory containing QM9 splits (default: %(default)s)')
    parser.add_argument('--isoform', type=int, default=[0, 0, 0, 0, 0], nargs='+', help='Number of atoms used for conditional generation H C N O F')
    args = parser.parse_args()

    output_file = os.path.join(os.path.dirname(args.xyz_file), "sample_quality.txt")
    with open(output_file, "a") as f:
        def log(s):
            print(s)
            f.write(s + "\n")

        # --- Compute synthetic molecule metrics ---
        atoms_synthetic = read(args.xyz_file, index=":")
        metrics_synthetic = ChemMetrics(atoms=atoms_synthetic)
        metrics_synthetic.compute_metrics()
        summary_synthetic = metrics_synthetic.summarize()

        # --- Load QM9 reference data ---
        atoms_QM9, summary_QM9, smiles_QM9 = load_QM9_split(
            data_dir=args.QM9_dir,
            split="train",
            load_info=True,
            load_smiles=True,
            as_ase_atoms=False
        )
        
        # --- Novelty ---
        summary_synthetic["novel_unique"] = len(set(metrics_synthetic.smiles) - set(smiles_QM9)) / len(atoms_synthetic)

        # --- Isoform matching ---
        if sum(args.isoform) != 0:
            isoforms = []
            for mol in atoms_synthetic:
                symbols = mol.get_chemical_symbols()
                counter = Counter(symbols)
                isoforms.append([
                    counter.get('H', 0),
                    counter.get('C', 0),
                    counter.get('N', 0),
                    counter.get('O', 0),
                    counter.get('F', 0),
                ])
            summary_synthetic["match_condition"] = sum([isoform_mol == args.isoform for isoform_mol in isoforms]) / len(atoms_synthetic)

        # --- Summary of global metrics ---
        log(f"Summary of synthetic molecules from {args.xyz_file}")
        log(f"Found {len(atoms_synthetic)} molecules")
        for key, value in summary_synthetic.items():
            if isinstance(value, float):
                log(f"{key:<20}{value:.5f}")

        # --- Summary table ---
        log(f"\n{'Atom type':<10}{'QM9(%)':>10}{'Syn(%)':>12}{'Counts':>12}{'Stable(%)':>14}")
        log("-" * 60)
        for atom in sorted(summary_QM9["atom_type_dist"].keys()):
            perc_qm9 = summary_QM9["atom_type_dist"][atom] * 100
            if atom not in summary_synthetic["atom_type_dist"].keys():
                perc_syn, counts_syn, stable_perc = 0, 0, 0
            else:
                perc_syn = summary_synthetic["atom_type_dist"][atom] * 100
                counts_syn = summary_synthetic["atom_type_hist"][atom] 
                stable_perc = (summary_synthetic["atom_stability_hist"][atom] / counts_syn) * 100
            log(f"{atom:<10}{perc_qm9:>10.2f}{perc_syn:>12.2f}{counts_syn:>12d}{stable_perc:>14.2f}")

    print(f"\nSample quality summary written to: {output_file}")

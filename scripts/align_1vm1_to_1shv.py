#!/usr/bin/env python3
"""
Superimpose 1VM1 (SHV-1 + tazobactam co-crystal) onto 1SHV (the apo
receptor this project's Stage 2 docking actually uses) via chain A CA
atoms, then apply the same rotation+translation to 1VM1's real TAZ
(intact tazobactam) ligand coordinates -- producing a crystal reference
for TAZ that is meaningful to compare against poses docked into 1SHV.

Unlike every other Stage 2 target (where receptor and crystal ligand
come from the SAME PDB entry, already in one coordinate frame), SHV-1's
receptor (1SHV, apo) and its real tazobactam co-crystal (1VM1) are two
independently-solved structures in two independent coordinate frames --
this alignment step is what every other target gets for free.
"""
from Bio.PDB import PDBParser, Superimposer
import numpy as np

STRUCT_DIR = "/home/kell/PathTwin/data/raw/structures"

parser = PDBParser(QUIET=True)
s_1shv = parser.get_structure("1SHV", f"{STRUCT_DIR}/1SHV.pdb")
s_1vm1 = parser.get_structure("1VM1", f"{STRUCT_DIR}/1VM1.pdb")

chain_1shv = s_1shv[0]["A"]
chain_1vm1 = s_1vm1[0]["A"]

# Match CA atoms by residue number (same protein, SHV-1 beta-lactamase --
# residue numbering should agree; only take residues present in both).
res_1shv = {r.id[1]: r for r in chain_1shv if r.id[0] == " " and "CA" in r}
res_1vm1 = {r.id[1]: r for r in chain_1vm1 if r.id[0] == " " and "CA" in r}
common = sorted(set(res_1shv) & set(res_1vm1))
print(f"Matched {len(common)} residues by number between 1SHV chain A and 1VM1 chain A")

fixed_atoms = [res_1shv[i]["CA"] for i in common]
moving_atoms = [res_1vm1[i]["CA"] for i in common]

sup = Superimposer()
sup.set_atoms(fixed_atoms, moving_atoms)
print(f"Superposition RMSD (CA, {len(common)} residues): {sup.rms:.3f} A")

# Apply the same transform to 1VM1's TAZ ligand atoms.
taz_atoms = []
for res in chain_1vm1:
    if res.resname == "TAZ":
        for atom in res:
            taz_atoms.append(atom)
print(f"TAZ atoms found: {len(taz_atoms)}")
sup.apply(taz_atoms)

out_path = f"{STRUCT_DIR}/1VM1_TAZ_A504_aligned_to_1SHV.pdb"
with open(out_path, "w") as f:
    for i, atom in enumerate(taz_atoms):
        coord = atom.coord
        f.write(
            f"HETATM{2085+i:5d} {atom.get_name():<4s} TAZ A 504    "
            f"{coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}  1.00  0.00           "
            f"{atom.element:>2s}\n"
        )
    f.write("END\n")
print(f"Wrote aligned TAZ crystal reference: {out_path}")

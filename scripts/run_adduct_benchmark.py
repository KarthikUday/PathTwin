#!/usr/bin/env python3
"""
PathTwin Stage 2 accuracy validation -- covalent-ADDUCT active-vs-decoy
docking re-run. Reuses run_accuracy_benchmark.py's docking/metrics
functions unchanged; only the (target_dir, box_file, receptor_rel,
active_rel, active_name) tuples differ -- same receptor PDBQT, same box,
same exhaustiveness(32)/seed(42) as each target's original free-drug run,
just a different active ligand (the adduct form) and a freshly generated,
adduct-matched decoy set (see generate_adduct_decoys.py).

This tests non-covalent Vina docking of the post-reaction PRODUCT SHAPE
as a proxy for covalent binding -- not true covalent docking (Vina has
no covalent-bond-formation term). A real methodological limitation,
stated plainly, not a claim about covalent-docking software this project
doesn't use.

Run from the project root (pathtwin-docking env):
  python3 scripts/run_adduct_benchmark.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_accuracy_benchmark import main as _shared_main  # noqa: E402
import run_accuracy_benchmark as rab  # noqa: E402

# Same box config + receptor as each target's original free-drug run;
# only the active ligand and decoy directory change.
ADDUCT_TARGETS = [
    ("shv1_tazobactam_adduct", "1shv_tazobactam.json",
     "data/raw/structures/1SHV_receptor.pdbqt",
     "data/raw/structures/TBE.pdbqt", "TBE"),
    ("oxa23_meropenem_adduct", "4jf4_meropenem.json",
     "data/raw/structures/4JF4_receptor.pdbqt",
     "data/raw/structures/MER.pdbqt", "MER"),
    ("pdc1_avibactam_adduct", "4hef_avibactam.json",
     "data/processed/docking/4hef_receptor.pdbqt",
     "data/raw/structures/NXL.pdbqt", "NXL"),
    ("pbp5_benzylpenicillin_adduct", "6mkg_benzylpenicillin.json",
     "data/processed/docking/6mkg_receptor.pdbqt",
     "data/raw/structures/PNM.pdbqt", "PNM"),
]

if __name__ == "__main__":
    rab.TARGETS = ADDUCT_TARGETS
    sys.argv = [sys.argv[0]]  # run all 4, no filtering
    _shared_main()

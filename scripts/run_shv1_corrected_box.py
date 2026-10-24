#!/usr/bin/env python3
"""
PathTwin Stage 2 accuracy validation -- SHV-1 box correction re-run.

Re-docks tazobactam + its existing 29 decoys (same molecules, same
PDBQTs already prepared for the original accuracy benchmark -- box
position doesn't affect which decoys were property-matched/selected)
through the SAME receptor (1SHV_receptor.pdbqt), exhaustiveness (32),
and seed (42) as the original run, but with the CORRECTED box center
(config/binding_boxes/1shv_tazobactam_corrected.json), taken from the
real crystallographic tazobactam position (1VM1, superimposed onto
1SHV) rather than the original Ser70-OG-centered box.

See DECISIONS_AND_LIMITATIONS.md "SHV-1 box correction" for the full
account of how the original box was found to be ~13 A off.

Run from the project root (pathtwin-docking env):
  python3 scripts/run_shv1_corrected_box.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_accuracy_benchmark import main as _shared_main  # noqa: E402
import run_accuracy_benchmark as rab  # noqa: E402

CORRECTED_TARGETS = [
    ("shv1_tazobactam_corrected", "1shv_tazobactam_corrected.json",
     "data/raw/structures/1SHV_receptor.pdbqt",
     "data/raw/structures/tazobactam.pdbqt", "tazobactam"),
]

if __name__ == "__main__":
    rab.TARGETS = CORRECTED_TARGETS
    sys.argv = [sys.argv[0]]
    _shared_main()

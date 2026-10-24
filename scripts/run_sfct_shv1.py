#!/usr/bin/env python3
"""
Rescore SHV-1's existing Vina-docked poses (1 active tazobactam + 29
decoys, from the accuracy benchmark) with OnionNet-SFCT, and rebuild the
active-vs-decoy AUC using the SFCT combined_score instead of raw Vina.

Same receptor/box/poses as the original benchmark -- this does NOT
re-dock anything, it rescores the poses Vina already produced.
"""
import glob
import json
import subprocess
import sys
from pathlib import Path

RECEPTOR = "/home/kell/PathTwin/data/raw/structures/1SHV_clean.pdb"
REF_CRYSTAL = "/home/kell/PathTwin/data/raw/structures/1VM1_TAZ_A504_aligned_to_1SHV.pdb"
DOCKED_DIR = "/home/kell/PathTwin/results/docking/accuracy_benchmark/shv1_tazobactam/docked"
SCORER = "/home/kell/OnionNet-SFCT/scorer.py"
MODEL = "/home/kell/OnionNet-SFCT/sfct.model"
OUT_DIR = "/home/kell/PathTwin/results/docking/sfct_rescoring/shv1_tazobactam"


def run_one(pdbqt_path: str, tag: str) -> dict:
    out_dat = f"{OUT_DIR}/{tag}_sfct.dat"
    cmd = [sys.executable, SCORER, "-r", RECEPTOR, "-l", pdbqt_path,
           "-o", out_dat, "--model", MODEL, "--ref", REF_CRYSTAL]
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, cwd=str(Path(SCORER).parent))
    if not Path(out_dat).exists():
        return {"tag": tag, "error": r.stderr[-500:] + r.stdout[-500:]}
    rows = []
    with open(out_dat) as f:
        header = f.readline()
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            rows.append({
                "name": parts[0], "pose_index": int(parts[1]),
                "origin_score": float(parts[2]), "combined_score": float(parts[3]),
                "sfct": float(parts[4]),
                "rmsd": (float(parts[5]) if len(parts) > 5 and parts[5] != "inf" else None),
            })
    if not rows:
        return {"tag": tag, "error": "no rows parsed", "raw_stderr": r.stderr[-300:]}
    # best pose by ORIGINAL vina score (pose 0 in vina's own numbering is
    # always its top pose already, but re-derive explicitly to be safe)
    best_by_vina = min(rows, key=lambda x: x["origin_score"])
    best_by_sfct = min(rows, key=lambda x: x["combined_score"])
    return {
        "tag": tag, "n_poses": len(rows),
        "best_by_vina_origin_score": best_by_vina["origin_score"],
        "best_by_vina_combined_score": best_by_vina["combined_score"],
        "best_by_sfct_pose_index": best_by_sfct["pose_index"],
        "best_by_sfct_combined_score": best_by_sfct["combined_score"],
        "best_by_sfct_origin_score": best_by_sfct["origin_score"],
        "same_top_pose": best_by_vina["pose_index"] == best_by_sfct["pose_index"],
        "rmsd_to_ref_best_vina_pose": best_by_vina["rmsd"],
        "rmsd_to_ref_best_sfct_pose": best_by_sfct["rmsd"],
    }


def main():
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    results = {}

    active_path = f"{DOCKED_DIR}/active_tazobactam_docked.pdbqt"
    print("Scoring active (tazobactam)...", flush=True)
    results["active"] = run_one(active_path, "active_tazobactam")
    print(" ", results["active"])

    decoy_files = sorted(glob.glob(f"{DOCKED_DIR}/decoy_*_docked.pdbqt"))
    for i, dp in enumerate(decoy_files):
        tag = Path(dp).stem.replace("_docked", "")
        res = run_one(dp, tag)
        results[tag] = res
        print(f"  [{i+1}/{len(decoy_files)}] {tag}: {res}", flush=True)

    with open(f"{OUT_DIR}/sfct_rescoring_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {OUT_DIR}/sfct_rescoring_summary.json")


if __name__ == "__main__":
    main()

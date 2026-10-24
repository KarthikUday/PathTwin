#!/usr/bin/env python3
"""
PathTwin Stage 2 accuracy validation -- active-vs-decoy docking benchmark.

For each of the 8 already-validated Stage 2 docking targets: docks the
target's own real active ligand PLUS its generated decoy set (see
generate_all_decoys.py / generate_decoys.py) against the identical
receptor/box/exhaustiveness/seed used for that target's original
validated run, then computes AUC-ROC and EF1% treating the active as the
sole positive and its decoys as negatives -- the standard DUD-E-style
active-vs-decoy evaluation, at this project's necessarily smaller scale
(one active + ~30 decoys per target, not DUD-E's ~224 actives + ~10,000
decoys per target -- see the honesty note this script writes into each
target's summary and DECISIONS_AND_LIMITATIONS.md for what that means for
interpreting AUC/EF1% here).

Run from the project root (pathtwin-docking env):
  python3 scripts/run_accuracy_benchmark.py
"""
import json
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BOX_DIR = PROJECT_ROOT / "config" / "binding_boxes"
BENCH_ROOT = PROJECT_ROOT / "results" / "docking" / "accuracy_benchmark"

# (target_dir, box_config, receptor_pdbqt, active_pdbqt, active_name)
TARGETS = [
    ("shv1_tazobactam", "1shv_tazobactam.json",
     "data/raw/structures/1SHV_receptor.pdbqt",
     "data/raw/structures/tazobactam.pdbqt", "tazobactam"),
    ("pbp3_jxj", "8gpw_jxj.json",
     "data/raw/structures/8GPW_receptor.pdbqt",
     "data/raw/structures/JXJ.pdbqt", "JXJ"),
    ("fosa_fosfomycin", "5v3d_fosfomycin.json",
     "data/raw/structures/5V3D_receptor.pdbqt",
     "data/raw/structures/fosfomycin.pdbqt", "fosfomycin"),
    ("pbp2a_cefepime", "5m18_cefepime.json",
     "data/raw/structures/5M18_receptor.pdbqt",
     "data/raw/structures/cefepime.pdbqt", "cefepime"),
    ("oxa23_meropenem", "4jf4_meropenem.json",
     "data/raw/structures/4JF4_receptor.pdbqt",
     "data/raw/structures/meropenem.pdbqt", "meropenem"),
    ("pdc1_avibactam", "4hef_avibactam.json",
     "data/processed/docking/4hef_receptor.pdbqt",
     "data/processed/docking/avibactam.pdbqt", "avibactam"),
    ("pbp5_benzylpenicillin", "6mkg_benzylpenicillin.json",
     "data/processed/docking/6mkg_receptor.pdbqt",
     "data/processed/docking/benzylpenicillin.pdbqt", "benzylpenicillin"),
    ("p99_cephalothin", "1bls_cephalothin.json",
     "data/processed/docking/1bls_receptor.pdbqt",
     "data/processed/docking/cephalothin.pdbqt", "cephalothin"),
]

VINA_RESULT_RE = re.compile(r"^REMARK VINA RESULT:\s*(-?\d+\.?\d*)", re.MULTILINE)


def prepare_ligand_pdbqt(sdf_path: Path, pdbqt_path: Path) -> bool:
    if pdbqt_path.exists():
        return True
    cmd = ["mk_prepare_ligand.py", "-i", str(sdf_path), "-o", str(pdbqt_path)]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)
    if r.returncode != 0 or not pdbqt_path.exists():
        print(f"    [warn] ligand prep failed for {sdf_path.name}: {r.stderr[-300:]}")
        return False
    return True


def run_vina(receptor: Path, ligand: Path, box: dict, out_pdbqt: Path, log_path: Path) -> float | None:
    if out_pdbqt.exists():
        # Resume support: a prior interrupted run may have already produced
        # this exact docked output. Reuse it instead of re-docking -- same
        # receptor/box/seed means the same command would reproduce the same
        # pose anyway, so re-running buys nothing but wall-clock time.
        cached = VINA_RESULT_RE.search(out_pdbqt.read_text(errors="ignore"))
        if cached:
            return float(cached.group(1))
    cmd = [
        "vina",
        "--receptor", str(receptor),
        "--ligand", str(ligand),
        "--center_x", str(box["center_x"]), "--center_y", str(box["center_y"]), "--center_z", str(box["center_z"]),
        "--size_x", str(box["size_x"]), "--size_y", str(box["size_y"]), "--size_z", str(box["size_z"]),
        "--exhaustiveness", str(box["exhaustiveness"]), "--seed", str(box["random_seed"]),
        "--out", str(out_pdbqt), "--num_modes", "9",
    ]
    with open(log_path, "w") as logf:
        r = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, cwd=PROJECT_ROOT)
    if r.returncode != 0 or not out_pdbqt.exists():
        return None
    text = out_pdbqt.read_text(errors="ignore")
    m = VINA_RESULT_RE.search(text)
    return float(m.group(1)) if m else None


def compute_metrics(active_affinity: float, decoy_affinities: list) -> dict:
    n_decoys = len(decoy_affinities)
    n_total = n_decoys + 1
    # rank ascending by affinity (more negative = better)
    worse = sum(1 for a in decoy_affinities if a > active_affinity)
    tied = sum(1 for a in decoy_affinities if a == active_affinity)
    auc = (worse + 0.5 * tied) / n_decoys if n_decoys else None
    active_rank = sum(1 for a in decoy_affinities if a < active_affinity) + 1
    percentile_better_than = round(100.0 * worse / n_decoys, 1) if n_decoys else None

    n_top1pct = max(1, round(0.01 * n_total))
    hit_top1pct = 1 if active_rank <= n_top1pct else 0
    ef1pct = round(hit_top1pct * n_total / n_top1pct, 2) if n_top1pct else None

    return {
        "n_decoys_scored": n_decoys,
        "n_total_pool": n_total,
        "active_affinity_kcal_mol": active_affinity,
        "active_rank": active_rank,
        "auc_single_active": round(auc, 3) if auc is not None else None,
        "percentile_better_than_decoys": percentile_better_than,
        "decoy_affinity_mean": round(sum(decoy_affinities) / n_decoys, 3) if n_decoys else None,
        "decoy_affinity_min": round(min(decoy_affinities), 3) if n_decoys else None,
        "decoy_affinity_max": round(max(decoy_affinities), 3) if n_decoys else None,
        "ef1pct": {
            "n_top_selected": n_top1pct,
            "active_in_top": bool(hit_top1pct),
            "ef1pct_value": ef1pct,
            "caveat": (f"n_top_selected={n_top1pct} molecule(s) out of a {n_total}-molecule pool "
                       f"(1 active + {n_decoys} decoys) -- this is the closest achievable "
                       f"approximation of a literal top-1% cutoff at this dataset size, not "
                       f"DUD-E's own thousands-of-decoys-per-target EF1%. With exactly 1 active, "
                       f"EF1% here is necessarily binary (either the single active is in that "
                       f"top slice or it isn't) rather than a smoothly-varying statistic."),
        },
    }


def main():
    only = set(sys.argv[1:]) if len(sys.argv) > 1 else None
    targets = [t for t in TARGETS if only is None or t[0] in only]
    overall_path = BENCH_ROOT / "overall_summary.json"
    overall = json.loads(overall_path.read_text()) if overall_path.exists() else {}
    for target_dir, box_file, receptor_rel, active_rel, active_name in targets:
        print(f"\n############ {target_dir} ############", flush=True)
        bench_dir = BENCH_ROOT / target_dir
        summary_path = bench_dir / "accuracy_summary.json"
        if summary_path.exists():
            # Resume support: this target already completed in a prior run
            # (interrupted before the whole batch finished). Reuse its
            # summary rather than re-docking ~30 ligands for nothing.
            print(f"  already complete ({summary_path}) -- reusing, not re-docking")
            overall[target_dir] = json.loads(summary_path.read_text())
            continue
        manifest_path = bench_dir / "decoy_manifest.json"
        if not manifest_path.exists():
            print(f"  [ERROR] no decoy manifest at {manifest_path}, skipping")
            continue
        manifest = json.loads(manifest_path.read_text())
        box = json.loads((BOX_DIR / box_file).read_text())
        receptor = PROJECT_ROOT / receptor_rel
        active_ligand = PROJECT_ROOT / active_rel
        if not receptor.exists() or not active_ligand.exists():
            print(f"  [ERROR] missing receptor/active pdbqt for {target_dir}")
            continue

        dock_dir = bench_dir / "docked"
        dock_dir.mkdir(exist_ok=True)

        # 1) active ligand -- fresh redock in this same batch
        print("  docking active ligand...", flush=True)
        active_affinity = run_vina(receptor, active_ligand, box,
                                    dock_dir / f"active_{active_name}_docked.pdbqt",
                                    dock_dir / f"active_{active_name}_vina_log.txt")
        if active_affinity is None:
            print(f"  [ERROR] active ligand docking failed for {target_dir}")
            continue
        print(f"    active affinity: {active_affinity} kcal/mol")

        # 2) prepare + dock each decoy
        decoy_results = []
        sdf_dir = bench_dir / "decoy_sdf"
        pdbqt_dir = bench_dir / "decoy_pdbqt"
        pdbqt_dir.mkdir(exist_ok=True)
        sdf_files = sorted(sdf_dir.glob("*.sdf"))
        for i, sdf_path in enumerate(sdf_files):
            pdbqt_path = pdbqt_dir / (sdf_path.stem + ".pdbqt")
            if not prepare_ligand_pdbqt(sdf_path, pdbqt_path):
                continue
            aff = run_vina(receptor, pdbqt_path, box,
                            dock_dir / f"{sdf_path.stem}_docked.pdbqt",
                            dock_dir / f"{sdf_path.stem}_vina_log.txt")
            if aff is None:
                print(f"    [warn] docking failed for {sdf_path.name}")
                continue
            decoy_results.append({"decoy_id": sdf_path.stem, "affinity_kcal_mol": aff})
            print(f"    [{i+1}/{len(sdf_files)}] {sdf_path.stem}: {aff} kcal/mol", flush=True)

        decoy_affinities = [d["affinity_kcal_mol"] for d in decoy_results]
        metrics = compute_metrics(active_affinity, decoy_affinities)
        summary = {
            "target": target_dir,
            "active_name": active_name,
            "n_decoys_generated": manifest["n_found"],
            "n_decoys_docked_successfully": len(decoy_results),
            "decoy_results": decoy_results,
            **metrics,
        }
        with open(bench_dir / "accuracy_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        overall[target_dir] = summary
        with open(overall_path, "w") as f:  # write incrementally, not just at the very end,
            json.dump(overall, f, indent=2)  # so a kill mid-batch doesn't lose finished targets
        print(f"  AUC(single-active)={metrics['auc_single_active']}  "
              f"rank={metrics['active_rank']}/{metrics['n_total_pool']}  "
              f"EF1%_hit={metrics['ef1pct']['active_in_top']}", flush=True)

    with open(BENCH_ROOT / "overall_summary.json", "w") as f:
        json.dump(overall, f, indent=2)
    print("\nALL_DOCKING_BENCHMARK_DONE", flush=True)


if __name__ == "__main__":
    main()

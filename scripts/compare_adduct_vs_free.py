#!/usr/bin/env python3
"""
PathTwin Stage 2 accuracy validation -- adduct-form vs. free-drug AUC
comparison for the 4 targets re-run with their real covalent/ring-opened
adduct form (see generate_adduct_decoys.py / run_adduct_benchmark.py).

For each of the 4 pairs, bootstraps both the adduct AUC and the free-drug
AUC (independently resampling each target's own decoy set, B iterations),
forms the distribution of (adduct AUC - free AUC), and reports its 95% CI
-- an unpaired two-sample comparison, since the two decoy sets are
independently generated (different molecule, different decoys). With 4
comparisons, both nominal (uncorrected) and Bonferroni-corrected
(alpha=0.05/4) significance are reported.

Run from the project root (any env with numpy is fine):
  python3 scripts/compare_adduct_vs_free.py
"""
import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BENCH_ROOT = PROJECT_ROOT / "results" / "docking" / "accuracy_benchmark"
B = 2000
SEED = 42
ALPHA = 0.05

PAIRS = [
    ("shv1_tazobactam", "shv1_tazobactam_adduct", "tazobactam", "TBE"),
    ("oxa23_meropenem", "oxa23_meropenem_adduct", "meropenem", "MER"),
    ("pdc1_avibactam", "pdc1_avibactam_adduct", "avibactam", "NXL"),
    ("pbp5_benzylpenicillin", "pbp5_benzylpenicillin_adduct", "benzylpenicillin", "PNM"),
]


def load_summary(target_dir: str):
    p = BENCH_ROOT / target_dir / "accuracy_summary.json"
    return json.loads(p.read_text())


def auc_from_affinities(active_affinity, decoy_affinities):
    n = len(decoy_affinities)
    worse = np.sum(decoy_affinities > active_affinity)
    tied = np.sum(decoy_affinities == active_affinity)
    return (worse + 0.5 * tied) / n


def bootstrap_auc_distribution(active_affinity, decoy_affinities, b, rng):
    n = len(decoy_affinities)
    idx = rng.integers(0, n, size=(b, n))
    resamples = decoy_affinities[idx]
    worse = np.sum(resamples > active_affinity, axis=1)
    tied = np.sum(resamples == active_affinity, axis=1)
    return (worse + 0.5 * tied) / n


def percentile_ci(dist, alpha):
    return float(np.percentile(dist, 100 * alpha / 2)), float(np.percentile(dist, 100 * (1 - alpha / 2)))


def main():
    rng = np.random.default_rng(SEED)
    bonferroni_alpha = ALPHA / len(PAIRS)
    print(f"{len(PAIRS)} adduct-vs-free comparisons. Nominal alpha=0.05; "
          f"Bonferroni-corrected alpha = {bonferroni_alpha:.4f}\n")

    results = []
    print(f"{'target':<26}{'free AUC':>10}{'adduct AUC':>12}{'diff':>8}{'95% CI (nominal)':>22}{'nominal':>10}{'Bonferroni':>12}")
    for free_dir, adduct_dir, free_name, adduct_name in PAIRS:
        free_s = load_summary(free_dir)
        adduct_s = load_summary(adduct_dir)
        free_active = free_s["active_affinity_kcal_mol"]
        free_decoys = np.array([d["affinity_kcal_mol"] for d in free_s["decoy_results"]])
        adduct_active = adduct_s["active_affinity_kcal_mol"]
        adduct_decoys = np.array([d["affinity_kcal_mol"] for d in adduct_s["decoy_results"]])

        free_point = auc_from_affinities(free_active, free_decoys)
        adduct_point = auc_from_affinities(adduct_active, adduct_decoys)
        free_dist = bootstrap_auc_distribution(free_active, free_decoys, B, rng)
        adduct_dist = bootstrap_auc_distribution(adduct_active, adduct_decoys, B, rng)
        diff_dist = adduct_dist - free_dist
        diff_point = adduct_point - free_point

        lo_nom, hi_nom = percentile_ci(diff_dist, ALPHA)
        lo_bonf, hi_bonf = percentile_ci(diff_dist, bonferroni_alpha)
        nominal_sig = bool(lo_nom > 0 or hi_nom < 0)
        bonf_sig = bool(lo_bonf > 0 or hi_bonf < 0)

        target_base = free_dir
        print(f"{target_base:<26}{free_point:>10.3f}{adduct_point:>12.3f}{diff_point:>+8.3f}"
              f"{'['+f'{lo_nom:+.3f}, {hi_nom:+.3f}'+']':>22}{('YES' if nominal_sig else 'no'):>10}"
              f"{('YES' if bonf_sig else 'no'):>12}")

        results.append({
            "target": target_base,
            "free_active_name": free_name, "adduct_active_name": adduct_name,
            "free_auc_point": round(free_point, 3), "adduct_auc_point": round(adduct_point, 3),
            "diff_point": round(diff_point, 3),
            "free_auc_ci95": [round(x, 3) for x in percentile_ci(free_dist, ALPHA)],
            "adduct_auc_ci95": [round(x, 3) for x in percentile_ci(adduct_dist, ALPHA)],
            "diff_ci95_nominal": [round(lo_nom, 3), round(hi_nom, 3)],
            "significant_nominal_95": nominal_sig,
            "diff_ci_bonferroni": [round(lo_bonf, 3), round(hi_bonf, 3)],
            "significant_bonferroni": bonf_sig,
            "verdict": ("adduct discriminates better" if (nominal_sig and diff_point > 0) else
                        "adduct discriminates worse" if (nominal_sig and diff_point < 0) else
                        "no distinguishable difference (within noise)"),
        })

    n_nom = sum(1 for r in results if r["significant_nominal_95"])
    n_bonf = sum(1 for r in results if r["significant_bonferroni"])
    print(f"\n{n_nom}/{len(PAIRS)} pairs distinguishable at nominal 95% (uncorrected); "
          f"{n_bonf}/{len(PAIRS)} survive Bonferroni correction for {len(PAIRS)} comparisons.\n")
    for r in results:
        print(f"  {r['target']}: {r['verdict']}")

    out = {
        "method": "percentile bootstrap, B=2000 per target, adduct AUC - free AUC, "
                  "unpaired (independently generated decoy sets)",
        "n_bootstrap_iterations": B,
        "seed": SEED,
        "n_comparisons": len(PAIRS),
        "bonferroni_alpha": round(bonferroni_alpha, 5),
        "comparisons": results,
        "n_significant_nominal_95": n_nom,
        "n_significant_bonferroni": n_bonf,
    }
    out_path = BENCH_ROOT / "adduct_vs_free_comparison.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()

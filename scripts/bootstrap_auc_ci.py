#!/usr/bin/env python3
"""
PathTwin Stage 2 accuracy validation -- bootstrap 95% CIs for each
target's active-vs-decoy AUC (see run_accuracy_benchmark.py for how the
point-estimate AUCs were computed).

Method: for each target, the active's docked affinity is fixed (it's a
single real measurement, not resampled); its N decoys' affinities are
resampled WITH REPLACEMENT (same size N) B times, and AUC is recomputed
against each resample and the standard bootstrap-the-negatives approach
for a single-positive ROC statistic. The resulting distribution of B
AUC values gives a percentile-method 95% CI per target.

For cross-target comparisons, an unpaired bootstrap: at each of the B
iterations, draw one resampled AUC for target i and one (independently
resampled) for target j, and form the distribution of AUC_i - AUC_j.
This is a genuine two-sample comparison (the two targets' decoy sets
are independently generated, not paired), not a paired test.

With 8 targets there are 28 pairwise comparisons, run at nominal 95%
CIs (no correction) AND flagged separately for whether they'd survive a
Bonferroni correction for 28 simultaneous comparisons (alpha=0.05/28),
so the multiple-comparisons risk is reported honestly rather than
silently ignored.

Run from the project root (any env with numpy is fine):
  python3 scripts/bootstrap_auc_ci.py
"""
import itertools
import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BENCH_ROOT = PROJECT_ROOT / "results" / "docking" / "accuracy_benchmark"
B = 2000
SEED = 42
ALPHA = 0.05
N_PAIRS = None  # filled in after target count is known


def auc_from_affinities(active_affinity: float, decoy_affinities: np.ndarray) -> float:
    """More-negative = better. AUC = fraction of decoys the active beats
    (i.e. active is more negative than), + 0.5 * ties, over n_decoys."""
    n = len(decoy_affinities)
    if n == 0:
        return float("nan")
    worse = np.sum(decoy_affinities > active_affinity)
    tied = np.sum(decoy_affinities == active_affinity)
    return (worse + 0.5 * tied) / n


def bootstrap_auc_distribution(active_affinity: float, decoy_affinities: np.ndarray,
                                b: int, rng: np.random.Generator) -> np.ndarray:
    n = len(decoy_affinities)
    idx = rng.integers(0, n, size=(b, n))
    resamples = decoy_affinities[idx]  # (b, n)
    worse = np.sum(resamples > active_affinity, axis=1)
    tied = np.sum(resamples == active_affinity, axis=1)
    return (worse + 0.5 * tied) / n


def percentile_ci(dist: np.ndarray, alpha: float = ALPHA):
    lo = np.percentile(dist, 100 * alpha / 2)
    hi = np.percentile(dist, 100 * (1 - alpha / 2))
    return float(lo), float(hi)


CANONICAL_8 = {
    "shv1_tazobactam", "pbp3_jxj", "fosa_fosfomycin", "pbp2a_cefepime",
    "oxa23_meropenem", "pdc1_avibactam", "pbp5_benzylpenicillin", "p99_cephalothin",
}


def main():
    # Scoped to the original 8 free-drug targets only -- kept stable even
    # as later re-runs (e.g. the covalent-adduct re-run, see
    # compare_adduct_vs_free.py) add more accuracy_summary.json files
    # under the same results/docking/accuracy_benchmark/ tree, so this
    # script's "8 targets / 28 pairwise comparisons" scope (matching what
    # README/DECISIONS_AND_LIMITATIONS.md already document) doesn't
    # silently balloon into meaningless cross-comparisons (e.g. one
    # target's free-drug AUC vs. a different target's adduct AUC).
    rng = np.random.default_rng(SEED)
    targets = {}
    for p in sorted(BENCH_ROOT.glob("*/accuracy_summary.json")):
        if p.parent.name not in CANONICAL_8:
            continue
        s = json.loads(p.read_text())
        active_aff = s["active_affinity_kcal_mol"]
        decoy_affs = np.array([d["affinity_kcal_mol"] for d in s["decoy_results"]])
        point_auc = auc_from_affinities(active_aff, decoy_affs)
        dist = bootstrap_auc_distribution(active_aff, decoy_affs, B, rng)
        lo, hi = percentile_ci(dist)
        targets[s["target"]] = {
            "active_name": s["active_name"],
            "n_decoys": len(decoy_affs),
            "point_auc": round(point_auc, 3),
            "bootstrap_mean": round(float(dist.mean()), 3),
            "ci95_lo": round(lo, 3),
            "ci95_hi": round(hi, 3),
            "distinguishable_from_chance": bool(lo > 0.5 or hi < 0.5),
            "_dist": dist,  # kept for pairwise comparisons below, stripped before saving
        }

    print(f"\n{'target':<24}{'active':<18}{'n':>4}{'AUC':>8}{'95% CI':>18}{'vs 0.5':>12}")
    for t, v in targets.items():
        ci_str = f"[{v['ci95_lo']:.3f}, {v['ci95_hi']:.3f}]"
        tag = "DISTINCT" if v["distinguishable_from_chance"] else "overlaps 0.5"
        print(f"{t:<24}{v['active_name']:<18}{v['n_decoys']:>4}{v['point_auc']:>8}{ci_str:>18}{tag:>16}")

    # pairwise comparisons
    names = list(targets.keys())
    pairs = list(itertools.combinations(names, 2))
    n_pairs = len(pairs)
    bonferroni_alpha = ALPHA / n_pairs
    print(f"\n{n_pairs} pairwise comparisons. Nominal alpha=0.05; "
          f"Bonferroni-corrected alpha for {n_pairs} comparisons = {bonferroni_alpha:.5f} "
          f"(i.e. a {100*(1-bonferroni_alpha):.3f}% CI needed to survive correction).")

    pairwise_results = []
    print(f"\n{'pair':<50}{'diff (i-j)':>12}{'95% CI (nominal)':>22}{'nominal':>10}{'Bonferroni':>12}")
    for i, j in pairs:
        di, dj = targets[i]["_dist"], targets[j]["_dist"]
        diff_dist = di - dj  # independent resamples, same iteration index -- fine, both are
        # i.i.d. draws from each target's own bootstrap distribution regardless of pairing
        diff_point = targets[i]["point_auc"] - targets[j]["point_auc"]
        lo_nom, hi_nom = percentile_ci(diff_dist, ALPHA)
        lo_bonf, hi_bonf = percentile_ci(diff_dist, bonferroni_alpha)
        nominal_sig = bool(lo_nom > 0 or hi_nom < 0)
        bonf_sig = bool(lo_bonf > 0 or hi_bonf < 0)
        pairwise_results.append({
            "target_i": i, "target_j": j, "diff_point": round(diff_point, 3),
            "ci95_nominal": [round(lo_nom, 3), round(hi_nom, 3)],
            "significant_nominal_95": nominal_sig,
            "ci_bonferroni": [round(lo_bonf, 3), round(hi_bonf, 3)],
            "significant_bonferroni": bonf_sig,
        })
        if nominal_sig:
            ci_str = f"[{lo_nom:+.3f}, {hi_nom:+.3f}]"
            print(f"{i+' vs '+j:<50}{diff_point:>+12.3f}{ci_str:>22}{'YES':>10}{('YES' if bonf_sig else 'no'):>12}")

    n_nominal_sig = sum(1 for r in pairwise_results if r["significant_nominal_95"])
    n_bonf_sig = sum(1 for r in pairwise_results if r["significant_bonferroni"])
    print(f"\n{n_nominal_sig}/{n_pairs} pairs distinguishable at nominal 95% CI "
          f"(uncorrected); {n_bonf_sig}/{n_pairs} survive Bonferroni correction for "
          f"{n_pairs} simultaneous comparisons.")

    # strip internal dist arrays before saving
    out = {}
    for t, v in targets.items():
        out[t] = {k: v[k] for k in v if k != "_dist"}
    result = {
        "method": "percentile bootstrap, B=2000, resample decoys with replacement per target, "
                  "active affinity held fixed",
        "n_bootstrap_iterations": B,
        "seed": SEED,
        "per_target": out,
        "n_pairwise_comparisons": n_pairs,
        "bonferroni_alpha": round(bonferroni_alpha, 6),
        "pairwise": pairwise_results,
        "n_pairs_significant_nominal_95": n_nominal_sig,
        "n_pairs_significant_bonferroni": n_bonf_sig,
    }
    out_path = BENCH_ROOT / "bootstrap_ci_summary.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()

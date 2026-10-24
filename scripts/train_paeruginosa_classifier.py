#!/usr/bin/env python3
"""
PathTwin Stage 5 extension -- P. aeruginosa fluoroquinolone resistance:
baseline (RGI gene presence/absence only) vs. combined (presence + five
regulatory/porin-locus SNP features) classifier comparison.

This is a direct methodological test, not just another species extension:
the literature on fluoroquinolone resistance in P. aeruginosa consistently
finds gene-PRESENCE features weak predictors here, because the dominant
resistance mechanisms in this species are regulatory derepression (mexR/
nalC/nalD -> MexAB-OprM; mexZ/nfxB -> MexXY) and porin loss (oprD) -- all
of which act through POINT MUTATIONS AND LOSS-OF-FUNCTION EVENTS in genes
that are present (and functionally intact) in essentially every P.
aeruginosa genome, WT or resistant alike. A presence/absence feature is
structurally blind to this: RGI reports "gene present" whether the copy is
wild-type or has a stop-gained mutation nine codons in. This script trains
the same RF + XGBoost pair on both feature sets, with the same grouped
(leakage-checked) split methodology used everywhere else in this project,
and reports the head-to-head result honestly either way.

v2 -- rerun at ~5.7x the first run's scale (1020 vs. 179 genomes) to test
whether the first run's real-but-inconclusive SHAP signal (mexZ_missense,
oprD_lof ranking high without a clear net accuracy/AUC win) firms up with
more data. XGBoost trains on GPU (`device="cuda"`) when one is actually
available -- this project's policy from here on for any tool that
supports it -- auto-probed at startup with a CPU fallback (`--no-gpu`
forces CPU outright) rather than assuming a GPU is present; Random
Forest has no GPU path in scikit-learn and runs on CPU regardless.

Inputs (from prep_paeruginosa_bvbrc_data.py + run_paeruginosa_pipeline.py):
  data/processed/paeruginosa_features_baseline.csv  -- RGI presence/absence only
  data/processed/paeruginosa_features_combined.csv  -- + 10 SNP features (5 genes x missense/lof)
  data/processed/paeruginosa_metadata.csv           -- genome_id, ST, bioproject_accession, is_resistant
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASELINE_CSV = PROJECT_ROOT / "data" / "processed" / "paeruginosa_features_baseline.csv"
DEFAULT_COMBINED_CSV = PROJECT_ROOT / "data" / "processed" / "paeruginosa_features_combined.csv"
DEFAULT_METADATA_CSV = PROJECT_ROOT / "data" / "processed" / "paeruginosa_metadata.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "ml_classifier"

ID_COL = "genome_id"
LABEL_COL = "is_resistant"
GROUP_COLS = {"st_grouped": "mlst", "bioproject_grouped": "bioproject_accession"}

RANDOM_STATE = 42
N_SPLITS = 5
N_TOP_SHAP_FEATURES = 15

SNP_FEATURE_NAMES = [
    f"{gene}_{kind}"
    for gene in ("mexR", "nalC", "nalD", "mexZ", "oprD")
    for kind in ("missense", "lof")
]


def _sanitize_feature_name(name: str) -> str:
    return name.replace("[", "(").replace("]", ")").replace("<", "")


def load_data(features_csv: Path, metadata_csv: Path):
    # genome_id values (e.g. "287.12451") look numeric and get silently
    # parsed as float64 otherwise, breaking the merge key against
    # metadata's str-typed genome_id (and losing trailing zeros in the
    # genome-id suffix in the process).
    features = pd.read_csv(features_csv, dtype={ID_COL: str})
    metadata = pd.read_csv(metadata_csv, dtype=str)
    metadata[LABEL_COL] = metadata[LABEL_COL].astype(int)

    # features CSV already carries is_resistant (written by
    # run_paeruginosa_pipeline.py from the same source), so only merge in
    # the grouping columns here.
    merged = features.merge(
        metadata[[ID_COL, "mlst", "bioproject_accession"]], on=ID_COL, how="inner"
    )
    if len(merged) != len(features):
        raise ValueError(
            f"Features/metadata row count mismatch after merge: "
            f"{len(features)} feature rows, {len(merged)} after merge."
        )

    feature_cols = [c for c in features.columns if c not in (ID_COL, LABEL_COL)]
    rename_map = {c: _sanitize_feature_name(c) for c in feature_cols if c != _sanitize_feature_name(c)}
    if rename_map:
        merged = merged.rename(columns=rename_map)
        feature_cols = [rename_map.get(c, c) for c in feature_cols]
        print(f"  Sanitized {len(rename_map)} feature name(s) for XGBoost compatibility")

    return merged, feature_cols


def _classification_metrics(y_true, y_pred, y_score) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    accuracy = float(np.mean(y_true == y_pred))
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred != 1) & (y_true == 1)))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "accuracy": accuracy, "precision": precision, "recall": recall,
        "f1": f1, "auc": float(roc_auc_score(y_true, y_score)),
    }


def make_grouped_split(merged: pd.DataFrame, group_col: str):
    df = merged.dropna(subset=[group_col]).copy()
    df = df[df[group_col].astype(str).str.strip() != ""]
    n_dropped = len(merged) - len(df)
    if n_dropped:
        print(f"  ({n_dropped} row(s) with missing '{group_col}' dropped)")

    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    train_idx, test_idx = next(sgkf.split(df, df[LABEL_COL], groups=df[group_col]))
    train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]

    overlap = set(train_df[group_col]) & set(test_df[group_col])
    if overlap:
        raise AssertionError(f"Group leakage in '{group_col}' split: {len(overlap)} group(s) shared")

    print(
        f"  train={len(train_df)} ({train_df[group_col].nunique()} unique {group_col} groups), "
        f"test={len(test_df)} ({test_df[group_col].nunique()} unique {group_col} groups), "
        f"0 groups shared (verified)"
    )
    return train_df, test_df


def resolve_xgb_device(force_cpu: bool) -> str:
    """This project's policy is to default to GPU for any tool that
    supports it, but a hardcoded device="cuda" crashes outright on any
    machine without a working CUDA GPU (a laptop, CI, a fresh clone of
    this repo) -- so probe for it instead of assuming it. `--no-gpu`
    short-circuits the probe and forces CPU outright; otherwise, fit a
    throwaway 2-row model with device="cuda" and fall back to CPU if
    that raises (no GPU present, drivers missing, or an XGBoost build
    without GPU support) rather than letting the real training run
    crash. Random Forest has no GPU path in scikit-learn and runs on
    CPU regardless, so this only affects XGBoost."""
    if force_cpu:
        print("XGBoost device: cpu (--no-gpu forced)")
        return "cpu"
    try:
        probe = XGBClassifier(n_estimators=2, device="cuda")
        probe.fit(np.zeros((4, 2)), np.array([0, 1, 0, 1]))
        print("XGBoost device: cuda (GPU probe succeeded)")
        return "cuda"
    except Exception as e:
        print(f"XGBoost device: cpu (GPU probe failed -- {e})")
        return "cpu"


def fit_and_evaluate_models(X_train, y_train, X_test, y_test, xgb_device: str):
    models = {
        # Random Forest has no GPU path in scikit-learn -- runs on CPU
        # regardless (this dataset's size makes that a non-issue: fitting
        # takes well under a second either way).
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
        # device: resolved once in main() via resolve_xgb_device() --
        # "cuda" when a working GPU was actually probed successfully,
        # "cpu" otherwise (--no-gpu, or the probe failed). Verified
        # directly that both .fit()/.predict() and shap.TreeExplainer()
        # work correctly against a GPU-trained XGBoost model in this
        # environment (an RTX 4090), and separately that the CPU path
        # completes end-to-end with --no-gpu forced (see STATUS.md).
        # (See _patch_shap_xgboost_base_score_bug() below for a real
        # XGBoost 3.x / shap 0.48 SHAP-generation compatibility bug this
        # project hit here -- pinning base_score does NOT avoid it,
        # verified directly; XGBoost re-serializes the field in bracketed
        # form regardless of value. Unrelated to the device setting.)
        "xgboost": XGBClassifier(n_estimators=200, random_state=RANDOM_STATE,
                                  eval_metric="logloss", device=xgb_device),
    }
    results, fitted = {}, {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        fitted[name] = model
        y_pred = model.predict(X_test)
        y_score = model.predict_proba(X_test)[:, 1]
        results[name] = _classification_metrics(y_test, y_pred, y_score)
    return results, fitted


def run_feature_set(label: str, merged: pd.DataFrame, feature_cols: list, xgb_device: str) -> dict:
    print(f"\n{'='*78}\n{label}  ({len(feature_cols)} features)\n{'='*78}")
    strategy_results, strategy_sizes, fitted_by_strategy = {}, {}, {}
    for strategy_name, group_col in GROUP_COLS.items():
        print(f"\n[{strategy_name}] StratifiedGroupKFold grouped on '{group_col}'...")
        train_df, test_df = make_grouped_split(merged, group_col)
        X_train, y_train = train_df[feature_cols], train_df[LABEL_COL]
        X_test, y_test = test_df[feature_cols], test_df[LABEL_COL]
        results, fitted = fit_and_evaluate_models(X_train, y_train, X_test, y_test, xgb_device)
        strategy_results[strategy_name] = results
        strategy_sizes[strategy_name] = {"n_train": len(train_df), "n_test": len(test_df)}
        fitted_by_strategy[strategy_name] = (fitted, X_test, y_test)
    return {"results": strategy_results, "sizes": strategy_sizes, "fitted": fitted_by_strategy}


def print_side_by_side(baseline_run: dict, combined_run: dict):
    print(f"\n{'='*90}")
    print("BASELINE (presence-only) vs COMBINED (presence + regulatory SNP features)")
    print(f"{'='*90}")
    for strategy_name in GROUP_COLS:
        print(f"\n--- {strategy_name} split ---")
        print(f"  {'Model':<15} {'Feature set':<12} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>8} {'AUC':>8}")
        print("  " + "-" * 80)
        for model_name in ["random_forest", "xgboost"]:
            b = baseline_run["results"][strategy_name][model_name]
            c = combined_run["results"][strategy_name][model_name]
            print(f"  {model_name:<15} {'baseline':<12} {b['accuracy']:>10.3f} {b['precision']:>10.3f} "
                  f"{b['recall']:>10.3f} {b['f1']:>8.3f} {b['auc']:>8.3f}")
            print(f"  {model_name:<15} {'combined':<12} {c['accuracy']:>10.3f} {c['precision']:>10.3f} "
                  f"{c['recall']:>10.3f} {c['f1']:>8.3f} {c['auc']:>8.3f}")
            d_acc = c['accuracy'] - b['accuracy']
            d_auc = c['auc'] - b['auc']
            print(f"  {'':>15} {'delta':<12} {d_acc:>+10.3f} {'':>10} {'':>10} {'':>8} {d_auc:>+8.3f}")


def _patch_shap_xgboost_base_score_bug():
    """XGBoost 3.x's raw UBJSON model dump serializes base_score as a
    bracketed array string (e.g. "[5E-1]"); shap 0.48's XGBTreeModelLoader
    calls bare float() on it and crashes ("could not convert string to
    float"). This is a real version-compatibility bug between the two
    libraries, not anything to do with our data or model config -- pinning
    XGBClassifier's own base_score explicitly does NOT avoid it, since
    XGBoost re-serializes the field in bracket form regardless of value
    (verified directly). Scoped monkeypatch: shap.explainers._tree's own
    `float` name is overridden to strip a leading/trailing "[]" before
    parsing -- a no-op for every normal float string already used
    elsewhere in that module, so this only changes behavior for the one
    buggy case."""
    import shap.explainers._tree as shap_tree_mod
    _real_float = float

    def _lenient_float(x):
        if isinstance(x, str):
            x = x.strip("[]")
        return _real_float(x)

    shap_tree_mod.float = _lenient_float


def generate_shap_summary(model, X_test, output_dir: Path, model_name: str, tag: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    if model_name == "xgboost":
        _patch_shap_xgboost_base_score_bug()

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    if isinstance(shap_values, list):
        shap_values = shap_values[-1]
    elif hasattr(shap_values, "ndim") and shap_values.ndim == 3:
        shap_values = shap_values[:, :, -1]

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    ranking = pd.Series(mean_abs_shap, index=X_test.columns).sort_values(ascending=False)
    top_features = ranking.head(N_TOP_SHAP_FEATURES)
    top_features_list = [{"feature": f, "mean_abs_shap": float(v), "rank": i + 1}
                          for i, (f, v) in enumerate(top_features.items())]

    snp_ranks = {f: int(np.where(ranking.index == f)[0][0]) + 1
                 for f in SNP_FEATURE_NAMES if f in ranking.index}

    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False, max_display=20)
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"paeruginosa_{tag}_shap_summary.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSHAP summary plot ({model_name}, {tag}) saved to: {out_path}")
    return out_path, top_features_list, snp_ranks, int(len(ranking))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-csv", default=str(DEFAULT_BASELINE_CSV))
    parser.add_argument("--combined-csv", default=str(DEFAULT_COMBINED_CSV))
    parser.add_argument("--metadata-csv", default=str(DEFAULT_METADATA_CSV))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--no-gpu", action="store_true", help="force XGBoost onto CPU")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    try:
        xgb_device = resolve_xgb_device(args.no_gpu)

        baseline_merged, baseline_cols = load_data(Path(args.baseline_csv), Path(args.metadata_csv))
        combined_merged, combined_cols = load_data(Path(args.combined_csv), Path(args.metadata_csv))

        n = len(combined_merged)
        n_resistant = int(combined_merged[LABEL_COL].sum())
        print(f"Loaded {n} genomes")
        print(f"Class balance: {n_resistant} Resistant ({n_resistant/n:.1%}) / "
              f"{n - n_resistant} Susceptible ({(n - n_resistant)/n:.1%})")
        print(f"Baseline feature count: {len(baseline_cols)} (RGI gene presence/absence)")
        print(f"Combined feature count: {len(combined_cols)} "
              f"({len(combined_cols) - len(baseline_cols)} SNP features added)")

        baseline_run = run_feature_set("BASELINE: RGI gene presence/absence only", baseline_merged, baseline_cols, xgb_device)
        combined_run = run_feature_set("COMBINED: presence/absence + 5-locus regulatory SNP features", combined_merged, combined_cols, xgb_device)

        print_side_by_side(baseline_run, combined_run)

        # SHAP on the combined model, primary (ST-grouped) split, better-AUC model
        primary = "st_grouped"
        fitted, X_test, y_test = combined_run["fitted"][primary]
        shap_model_name = max(
            combined_run["results"][primary], key=lambda m: combined_run["results"][primary][m]["auc"]
        )
        shap_path, top_shap_features, snp_ranks, n_total_features = generate_shap_summary(
            fitted[shap_model_name], X_test, output_dir, shap_model_name, "combined"
        )

        print(f"\nTop {N_TOP_SHAP_FEATURES} SHAP features (combined model, {shap_model_name}, {primary} split):")
        for item in top_shap_features:
            marker = "  <-- SNP feature" if item["feature"] in SNP_FEATURE_NAMES else ""
            print(f"  {item['rank']:2d}. {item['feature']:<30} {item['mean_abs_shap']:.4f}{marker}")

        print(f"\nSNP feature ranks (out of {n_total_features} total features, combined model):")
        for f in SNP_FEATURE_NAMES:
            rank = snp_ranks.get(f, None)
            print(f"  {f:<20} rank={'not present / zero importance' if rank is None else rank}")

        snp_in_top = [f for f in SNP_FEATURE_NAMES if snp_ranks.get(f, 9999) <= N_TOP_SHAP_FEATURES]
        if snp_in_top:
            print(f"\n  *** {len(snp_in_top)} SNP feature(s) rank in the top {N_TOP_SHAP_FEATURES}: {snp_in_top} ***")
        else:
            print(f"\n  *** No SNP feature ranks in the top {N_TOP_SHAP_FEATURES} -- "
                  f"the SNP-feature hypothesis is NOT supported by SHAP importance ***")

        summary = {
            "n_genomes": n,
            "n_resistant": n_resistant,
            "n_susceptible": n - n_resistant,
            "n_baseline_features": len(baseline_cols),
            "n_combined_features": len(combined_cols),
            "n_snp_features_added": len(combined_cols) - len(baseline_cols),
            "baseline_metrics": baseline_run["results"],
            "combined_metrics": combined_run["results"],
            "baseline_split_sizes": baseline_run["sizes"],
            "combined_split_sizes": combined_run["sizes"],
            "shap_primary_split": primary,
            "shap_model_used": shap_model_name,
            "shap_top_features": top_shap_features,
            "snp_feature_shap_ranks": snp_ranks,
            "snp_features_in_top_n": snp_in_top,
            "shap_summary_plot": str(shap_path),
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        results_path = output_dir / "paeruginosa_fq_stage5_results.json"
        with open(results_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nResults summary saved to: {results_path}")

    except Exception as e:
        sys.exit(f"ERROR: {e}")


if __name__ == "__main__":
    main()

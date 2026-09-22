#!/usr/bin/env python3
"""
PathTwin Stage 5 extension -- generic BV-BRC AMR classifier trainer, used
identically for the three new ESKAPE species extensions (A. baumannii/
carbapenem, E. faecium/vancomycin, E. cloacae/ESC-cephalosporin) rather
than three near-duplicate per-species scripts -- the same "one script,
many species" precedent this project already established with
crispr_scan.py (Stage 6, run unmodified across all 6 ESKAPE pathogens).

Reuses the KlebNET-GSP classifier's leak-proof methodology exactly as
S. aureus/P. aeruginosa's v2 extensions did: grouped splits (ST and
BioProject) are the ONLY evaluation from the start -- no random-split
round-trip needed, unlike the original K. pneumoniae classifier's first
pass. XGBoost trains on GPU (device="cuda") per this project's
default-to-GPU policy; Random Forest has no GPU path in scikit-learn and
runs on CPU regardless. Includes the same shap 0.48 / XGBoost 3.x
base_score monkeypatch already needed for the P. aeruginosa extension.

Input CSVs match every other Stage 5 BV-BRC extension's prep-script
output: `<features-csv>` (strain, is_resistant, N raw sp_gene features),
`<metadata-csv>` (strain, ST, bioproject_accession, genome_name).
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from xgboost import XGBClassifier

STRAIN_COL = "strain"
LABEL_COL = "is_resistant"
GROUP_COLS = {"st_grouped": "ST", "bioproject_grouped": "bioproject_accession"}

RANDOM_STATE = 42
N_SPLITS = 5
N_TOP_SHAP_FEATURES = 15


def _sanitize_feature_name(name: str) -> str:
    """XGBoost rejects feature names containing '[', ']', or '<' -- some
    sp_gene product-name-derived feature columns have them."""
    return name.replace("[", "(").replace("]", ")").replace("<", "")


def load_data(features_csv, metadata_csv):
    # dtype=str on the merge key: BV-BRC genome IDs look numeric
    # ("470.12345") and pandas' default float64 inference can silently
    # collide distinct IDs -- the exact bug caught and fixed in this
    # project's Stage 6 v2 CRISPR/AMR analysis script.
    features = pd.read_csv(features_csv, dtype={STRAIN_COL: str})
    metadata = pd.read_csv(metadata_csv, dtype={STRAIN_COL: str})
    merged = features.merge(metadata, on=STRAIN_COL, how="inner")
    if len(merged) != len(features):
        raise ValueError(
            f"Features/metadata row count mismatch after merge: "
            f"{len(features)} features rows, {len(merged)} after merge."
        )
    feature_cols = [c for c in features.columns if c not in (STRAIN_COL, LABEL_COL)]

    rename_map = {c: _sanitize_feature_name(c) for c in feature_cols if c != _sanitize_feature_name(c)}
    if rename_map:
        merged = merged.rename(columns=rename_map)
        feature_cols = [rename_map.get(c, c) for c in feature_cols]
        print(f"Sanitized {len(rename_map)} feature name(s) for XGBoost compatibility")

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


# A single group holding more than this fraction of the whole (post-drop)
# dataset makes a 5-fold split's size balance unreliable -- one oversized
# group forces whichever fold it lands in to dominate, starving the other
# folds. Caught concretely in this project already (P. aeruginosa's
# BioProject-grouped split: one group was 57% of the pool, produced a
# degenerate 1.000-accuracy/NaN-AUC fold) -- this constant + the
# diagnostics below generalize that catch into the tool itself instead of
# re-discovering it by eye for every new species.
DOMINANT_GROUP_WARN_FRACTION = 0.15
MIN_RELIABLE_TEST_FRACTION = 0.5  # vs. the 1/N_SPLITS a balanced split would give


def make_grouped_split(merged: pd.DataFrame, group_col: str):
    """Returns (train_df, test_df, diagnostics). diagnostics always
    includes dominant_group_frac and split_reliable so a degenerate split
    (tiny/skewed test set from one oversized group) is flagged
    automatically rather than only caught by eyeballing a suspiciously
    perfect accuracy after the fact."""
    df = merged.dropna(subset=[group_col]).copy()
    df = df[df[group_col].astype(str).str.strip() != ""]
    n_dropped = len(merged) - len(df)
    if n_dropped:
        print(f"  ({n_dropped} row(s) with missing '{group_col}' dropped -- can't be grouped)")

    group_sizes = df[group_col].value_counts()
    dominant_group_frac = float(group_sizes.iloc[0] / len(df)) if len(group_sizes) else float("nan")
    if dominant_group_frac > DOMINANT_GROUP_WARN_FRACTION:
        print(f"  *** WARNING: largest '{group_col}' group ({group_sizes.index[0]!r}) is "
              f"{dominant_group_frac:.1%} of the dataset -- a 5-fold split's size balance "
              f"is likely unreliable regardless of the resulting accuracy number ***")

    # Minority class must have at least N_SPLITS members for
    # StratifiedGroupKFold to even construct the requested number of
    # folds -- reduce n_splits rather than let it hard-crash, and report
    # the reduction rather than silently swallowing it.
    class_counts = df[LABEL_COL].value_counts()
    min_class_n = int(class_counts.min()) if len(class_counts) else 0
    n_splits = N_SPLITS
    if min_class_n < 2:
        return None, None, {
            "dominant_group_frac": dominant_group_frac, "split_reliable": False,
            "reason": f"minority class has only {min_class_n} example(s) after dropping "
                      f"missing-'{group_col}' rows -- no split of any kind is computable",
        }
    if min_class_n < N_SPLITS:
        n_splits = min_class_n
        print(f"  (minority class has only {min_class_n} example(s) -- reducing n_splits "
              f"from {N_SPLITS} to {n_splits} so StratifiedGroupKFold can run at all; "
              f"the resulting test set will still be tiny, read the metrics accordingly)")

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    train_idx, test_idx = next(sgkf.split(df, df[LABEL_COL], groups=df[group_col]))
    train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]

    overlap = set(train_df[group_col]) & set(test_df[group_col])
    if overlap:
        raise AssertionError(f"Group leakage in '{group_col}' split: {len(overlap)} group(s) shared")

    expected_test_frac = 1.0 / n_splits
    actual_test_frac = len(test_df) / len(df)
    size_ok = actual_test_frac >= MIN_RELIABLE_TEST_FRACTION * expected_test_frac
    test_has_both_classes = test_df[LABEL_COL].nunique() >= 2
    split_reliable = (dominant_group_frac <= DOMINANT_GROUP_WARN_FRACTION) and size_ok and test_has_both_classes

    print(
        f"  train={len(train_df)} ({train_df[group_col].nunique()} unique {group_col} groups), "
        f"test={len(test_df)} ({test_df[group_col].nunique()} unique {group_col} groups), "
        f"0 groups shared (verified)"
    )
    if not split_reliable:
        reasons = []
        if dominant_group_frac > DOMINANT_GROUP_WARN_FRACTION:
            reasons.append(f"dominant group {dominant_group_frac:.1%} of data")
        if not size_ok:
            reasons.append(f"test set only {actual_test_frac:.1%} of data (expected ~{expected_test_frac:.1%})")
        if not test_has_both_classes:
            reasons.append("test set has only one class present")
        print(f"  *** split flagged UNRELIABLE: {'; '.join(reasons)} -- "
              f"treat this split's metrics as uninterpretable, not a headline result ***")

    diagnostics = {
        "dominant_group_frac": dominant_group_frac,
        "dominant_group": str(group_sizes.index[0]) if len(group_sizes) else None,
        "n_splits_used": n_splits,
        "actual_test_fraction": actual_test_frac,
        "test_has_both_classes": bool(test_has_both_classes),
        "split_reliable": bool(split_reliable),
    }
    return train_df, test_df, diagnostics


def _patch_shap_xgboost_base_score_bug():
    """XGBoost 3.x's raw UBJSON model dump serializes base_score as a
    bracketed array string (e.g. "[5E-1]"); shap 0.48's XGBTreeModelLoader
    calls bare float() on it and crashes. Scoped monkeypatch, same fix
    already applied in train_paeruginosa_classifier.py -- see that file
    for the full explanation of why pinning base_score does not avoid it."""
    import shap.explainers._tree as shap_tree_mod
    _real_float = float

    def _lenient_float(x):
        if isinstance(x, str):
            x = x.strip("[]")
        return _real_float(x)

    shap_tree_mod.float = _lenient_float


def fit_and_evaluate_models(X_train, y_train, X_test, y_test, use_gpu: bool) -> dict:
    models = {
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
        "xgboost": XGBClassifier(
            n_estimators=200, random_state=RANDOM_STATE, eval_metric="logloss",
            device="cuda" if use_gpu else "cpu",
        ),
    }
    results, fitted = {}, {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        fitted[name] = model
        y_pred = model.predict(X_test)
        y_score = model.predict_proba(X_test)[:, 1]
        results[name] = _classification_metrics(y_test, y_pred, y_score)
    return results, fitted


def print_comparison_table(strategy_results: dict):
    print(f"\n{'='*78}")
    print("GROUPED-SPLIT RESULTS (both are leakage-checked -- neither is a naive random baseline)")
    print(f"{'='*78}")
    for model_name in ["random_forest", "xgboost"]:
        print(f"\n{model_name}")
        print(f"  {'Split':<20} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>8} {'AUC':>8}")
        print("  " + "-" * 70)
        for strategy_name, per_model in strategy_results.items():
            m = per_model[model_name]
            print(
                f"  {strategy_name:<20} {m['accuracy']:>10.3f} {m['precision']:>10.3f} "
                f"{m['recall']:>10.3f} {m['f1']:>8.3f} {m['auc']:>8.3f}"
            )


def generate_shap_summary(model, model_name, X_test, output_dir: Path, tag: str):
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

    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False, max_display=20)
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{tag}_shap_summary.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSHAP summary plot ({model_name}, {tag}) saved to: {out_path}")
    return out_path, top_features_list, int(len(ranking))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-csv", required=True)
    parser.add_argument("--metadata-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tag", required=True, help="short species/phenotype tag for output filenames, e.g. 'abaumannii_carbapenem'")
    parser.add_argument("--expected-markers", default="",
                         help="comma-separated substrings (case-insensitive) of feature names expected to "
                              "dominate SHAP for this phenotype, e.g. 'OXA-23,OXA-24,OXA-40,OXA-58,ISAba1'")
    parser.add_argument("--no-gpu", action="store_true", help="force XGBoost onto CPU")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    expected_markers = [m.strip() for m in args.expected_markers.split(",") if m.strip()]

    try:
        merged, feature_cols = load_data(args.features_csv, args.metadata_csv)
        n = len(merged)
        n_resistant = int(merged[LABEL_COL].sum())
        print(f"Loaded {n} strains x {len(feature_cols)} features")
        print(f"Class balance: {n_resistant} Resistant ({n_resistant/n:.1%}) / "
              f"{n - n_resistant} Susceptible ({(n - n_resistant)/n:.1%})")

        strategy_results, strategy_sizes, fitted_by_strategy = {}, {}, {}
        for strategy_name, group_col in GROUP_COLS.items():
            print(f"\n[{strategy_name}] StratifiedGroupKFold grouped on '{group_col}'...")
            train_df, test_df, diag = make_grouped_split(merged, group_col)
            if train_df is None:
                print(f"  *** split NOT COMPUTABLE: {diag['reason']} ***")
                strategy_results[strategy_name] = None
                strategy_sizes[strategy_name] = diag
                continue
            X_train, y_train = train_df[feature_cols], train_df[LABEL_COL]
            X_test, y_test = test_df[feature_cols], test_df[LABEL_COL]
            results, fitted = fit_and_evaluate_models(X_train, y_train, X_test, y_test, use_gpu=not args.no_gpu)
            strategy_results[strategy_name] = results
            strategy_sizes[strategy_name] = {
                "n_train": len(train_df), "n_test": len(test_df),
                "n_train_groups": int(train_df[group_col].nunique()),
                "n_test_groups": int(test_df[group_col].nunique()),
                **diag,
            }
            fitted_by_strategy[strategy_name] = (fitted, X_test)

        computable = {k: v for k, v in strategy_results.items() if v is not None}
        print_comparison_table(computable) if computable else print("\n*** No grouped split was computable for either grouping ***")

        # SHAP source: prefer a RELIABLE split (st_grouped first, matching
        # every other extension's primary-split convention), fall back to
        # any computable split if none are reliable, fall back further to
        # fitting on the full dataset with no held-out set at all if
        # neither grouping could even be split -- always labeled clearly
        # in the output rather than silently presented as equivalent to a
        # real held-out evaluation.
        ordered_candidates = [s for s in ("st_grouped", "bioproject_grouped") if s in computable]
        reliable_candidates = [s for s in ordered_candidates if strategy_sizes[s].get("split_reliable")]
        if reliable_candidates:
            primary = reliable_candidates[0]
            shap_source_note = f"reliable {primary} split"
        elif ordered_candidates:
            primary = ordered_candidates[0]
            shap_source_note = f"{primary} split (flagged UNRELIABLE -- SHAP ranking still informative, metrics are not)"
        else:
            primary = None
            shap_source_note = "no grouped split was computable -- fit on the FULL dataset, no held-out test set"

        if primary is not None:
            fitted, X_test = fitted_by_strategy[primary]
            shap_model_name = max(strategy_results[primary], key=lambda m: strategy_results[primary][m]["auc"])
        else:
            # Full-data fallback: fit fresh models on everything purely to
            # get a SHAP ranking -- there is no accuracy/AUC to report
            # here since there's no held-out set, and none is claimed.
            X_full, y_full = merged[feature_cols], merged[LABEL_COL]
            _, fitted = fit_and_evaluate_models(X_full, y_full, X_full, y_full, use_gpu=not args.no_gpu)
            X_test = X_full
            shap_model_name = "random_forest"  # arbitrary but deterministic; no AUC to pick by here
        print(f"\nSHAP source: {shap_source_note}")
        shap_path, top_shap_features, n_total_features = generate_shap_summary(
            fitted[shap_model_name], shap_model_name, X_test, output_dir, args.tag
        )

        print(f"\nTop {N_TOP_SHAP_FEATURES} SHAP features ({shap_model_name}, {primary} split, "
              f"out of {n_total_features} total features):")
        for item in top_shap_features:
            print(f"  {item['rank']:2d}. {item['feature']:<55} {item['mean_abs_shap']:.4f}")

        # Word-boundary regex, not naive substring containment: a naive
        # `marker.lower() in fname.lower()` check let the short marker
        # "ACT" (the ACT/MIR AmpC beta-lactamase family) match inside the
        # ordinary English word "factor" ("...elongation fACTor G..."),
        # producing a false "expected mechanism found" hit for E. cloacae
        # that had nothing to do with AmpC. Caught by actually reading the
        # matched feature names rather than trusting the count. \b on
        # both sides rejects that (no word-char/non-word-char transition
        # inside "factor") while still matching real hits like "ACT-12"
        # or "ACT/MIR family" (hyphen/slash/space are non-word chars).
        expected_hits = []
        if expected_markers:
            marker_patterns = [(m, re.compile(rf"\b{re.escape(m)}\b", re.IGNORECASE)) for m in expected_markers]
            for item in top_shap_features:
                for marker, pattern in marker_patterns:
                    if pattern.search(item["feature"]):
                        expected_hits.append({"feature": item["feature"], "rank": item["rank"], "marker": marker})
                        break
            if expected_hits:
                top1_is_expected = top_shap_features[0]["feature"] == expected_hits[0]["feature"] and expected_hits[0]["rank"] == 1
                if top1_is_expected:
                    print(f"\n  *** expected-mechanism marker is the #1 SHAP feature -- "
                          f"clean biological validation ***")
                else:
                    print(f"\n  *** {len(expected_hits)} expected-mechanism marker(s) in top "
                          f"{N_TOP_SHAP_FEATURES} (not ranked #1): "
                          f"{[h['feature'] for h in expected_hits]} ***")
            else:
                print(f"\n  *** WARNING: none of the expected markers ({expected_markers}) appear in the "
                      f"top {N_TOP_SHAP_FEATURES} SHAP features -- expected mechanism NOT confirmed dominant "
                      f"by this model/split ***")

        summary = {
            "tag": args.tag,
            "n_genomes": n,
            "n_resistant": n_resistant,
            "n_susceptible": n - n_resistant,
            "n_features": len(feature_cols),
            "split_sizes": strategy_sizes,
            "metrics": strategy_results,
            "shap_primary_split": primary,
            "shap_source_note": shap_source_note,
            "shap_model_used": shap_model_name,
            "shap_top_features": top_shap_features,
            "expected_markers": expected_markers,
            "expected_marker_hits_in_top_n": expected_hits,
            "shap_summary_plot": str(shap_path),
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        results_path = output_dir / f"{args.tag}_stage5_results.json"
        with open(results_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nResults summary saved to: {results_path}")

    except Exception as e:
        sys.exit(f"ERROR: {e}")


if __name__ == "__main__":
    main()

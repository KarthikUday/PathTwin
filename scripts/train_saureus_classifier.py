#!/usr/bin/env python3
"""
PathTwin Stage 5 extension -- S. aureus oxacillin resistance classifier

Loads scripts/prep_saureus_bvbrc_data.py's output
(data/processed/saureus_features.csv: strain, is_resistant, 329 raw
sp_gene antibiotic-resistance features; saureus_metadata.csv: strain, ST,
bioproject_accession), trains Random Forest + XGBoost, and generates a
SHAP summary -- reusing the KlebNET-GSP classifier's leak-proof
methodology, but applying the lesson learned there from the start rather
than needing a second pass: grouped splits (StratifiedGroupKFold on ST,
and separately on BioProject) are the ONLY evaluation here. There is no
random-split baseline to compare against and no follow-up leakage check
needed -- this IS the leakage-checked result.

Both groupings are run (not just one) for the same reason KlebNET's
grouped-split check used two: ST catches clonal leakage (near-identical
genomes on both sides of a split), BioProject catches study/batch leakage
(closer to real-world external validation, since a deployed classifier
will see genomes from studies it never trained on).
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
DEFAULT_FEATURES_CSV = PROJECT_ROOT / "data" / "processed" / "saureus_features.csv"
DEFAULT_METADATA_CSV = PROJECT_ROOT / "data" / "processed" / "saureus_metadata.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "ml_classifier"

STRAIN_COL = "strain"
LABEL_COL = "is_resistant"
GROUP_COLS = {"st_grouped": "ST", "bioproject_grouped": "bioproject_accession"}

RANDOM_STATE = 42
N_SPLITS = 5
N_TOP_SHAP_FEATURES = 10


def _sanitize_feature_name(name: str) -> str:
    """XGBoost rejects feature names containing '[', ']', or '<' -- a
    handful of sp_gene product-name-derived feature columns have them
    (e.g. "...[acyl-carrier-protein]..."). Strip just those characters;
    keeps names human-readable and doesn't touch the ~324 columns that
    don't need it."""
    return name.replace("[", "(").replace("]", ")").replace("<", "")


def load_data(features_csv, metadata_csv):
    features = pd.read_csv(features_csv)
    metadata = pd.read_csv(metadata_csv)
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
        print(f"Sanitized {len(rename_map)} feature name(s) for XGBoost compatibility "
              f"(stripped [, ], < -- e.g. {next(iter(rename_map.values()))!r})")

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
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": float(roc_auc_score(y_true, y_score)),
    }


def make_grouped_split(merged: pd.DataFrame, group_col: str):
    """
    StratifiedGroupKFold(n_splits=5), first fold as the held-out test set
    (~20%, approximately -- exact size depends on group sizes). Rows with
    a missing group_col value are dropped first.
    """
    df = merged.dropna(subset=[group_col]).copy()
    df = df[df[group_col].astype(str).str.strip() != ""]
    n_dropped = len(merged) - len(df)
    if n_dropped:
        print(f"  ({n_dropped} row(s) with missing '{group_col}' dropped -- can't be grouped)")

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


def fit_and_evaluate_models(X_train, y_train, X_test, y_test) -> dict:
    models = {
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
        "xgboost": XGBClassifier(n_estimators=200, random_state=RANDOM_STATE, eval_metric="logloss"),
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


def generate_shap_summary(model, X_test, output_dir: Path, model_name: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    if isinstance(shap_values, list):
        shap_values = shap_values[-1]
    elif hasattr(shap_values, "ndim") and shap_values.ndim == 3:
        shap_values = shap_values[:, :, -1]

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_features = (
        pd.Series(mean_abs_shap, index=X_test.columns)
        .sort_values(ascending=False)
        .head(N_TOP_SHAP_FEATURES)
    )
    top_features_list = [{"feature": f, "mean_abs_shap": float(v)} for f, v in top_features.items()]

    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "saureus_shap_summary.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSHAP summary plot ({model_name}) saved to: {out_path}")
    return out_path, top_features_list


def main():
    parser = argparse.ArgumentParser(
        description="Train RF + XGBoost on S. aureus oxacillin resistance (BV-BRC), grouped splits only."
    )
    parser.add_argument("--features-csv", default=str(DEFAULT_FEATURES_CSV))
    parser.add_argument("--metadata-csv", default=str(DEFAULT_METADATA_CSV))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

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
            train_df, test_df = make_grouped_split(merged, group_col)
            X_train, y_train = train_df[feature_cols], train_df[LABEL_COL]
            X_test, y_test = test_df[feature_cols], test_df[LABEL_COL]
            results, fitted = fit_and_evaluate_models(X_train, y_train, X_test, y_test)
            strategy_results[strategy_name] = results
            strategy_sizes[strategy_name] = {"n_train": len(train_df), "n_test": len(test_df)}
            fitted_by_strategy[strategy_name] = (fitted, X_test)

        print_comparison_table(strategy_results)

        # SHAP from the primary (ST-grouped) split's better-AUC model.
        primary = "st_grouped"
        fitted, X_test = fitted_by_strategy[primary]
        shap_model_name = max(strategy_results[primary], key=lambda m: strategy_results[primary][m]["auc"])
        shap_path, top_shap_features = generate_shap_summary(
            fitted[shap_model_name], X_test, output_dir, shap_model_name
        )

        print(f"\nTop {N_TOP_SHAP_FEATURES} SHAP features ({shap_model_name}, {primary} split):")
        for i, item in enumerate(top_shap_features, 1):
            print(f"  {i:2d}. {item['feature']:<55} {item['mean_abs_shap']:.4f}")
        mec_related = [f for f in top_shap_features if "mec" in f["feature"].lower()]
        if mec_related and top_shap_features[0] is mec_related[0]:
            print(f"\n  *** mec-family gene is the #1 SHAP feature -- expected biological validation ***")
        elif mec_related:
            print(f"\n  *** mec-family gene(s) present in top {N_TOP_SHAP_FEATURES}, not ranked #1 ***")
        else:
            print(f"\n  *** WARNING: no mec-family gene in top {N_TOP_SHAP_FEATURES} SHAP features ***")

        summary = {
            "n_genomes": n,
            "n_resistant": n_resistant,
            "n_susceptible": n - n_resistant,
            "n_features": len(feature_cols),
            "split_sizes": strategy_sizes,
            "metrics": strategy_results,
            "shap_primary_split": primary,
            "shap_model_used": shap_model_name,
            "shap_top_features": top_shap_features,
            "shap_summary_plot": str(shap_path),
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        results_path = output_dir / "saureus_stage5_results.json"
        with open(results_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nResults summary saved to: {results_path}")

    except Exception as e:
        sys.exit(f"ERROR: {e}")


if __name__ == "__main__":
    main()

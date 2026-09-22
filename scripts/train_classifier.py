#!/usr/bin/env python3
"""
PathTwin Stage 5 — resistance-phenotype classifier training (KlebNET-GSP)

Loads the KlebNET-GSP feature matrix + label (scripts/prep_klebnet_data.py's
data/processed/klebnet_features.csv: strain, is_resistant, 42 individual
QRDR/PMQR allele + oqxA_copy/oqxB_copy features) and its metadata/benchmark
CSV (data/processed/klebnet_metadata.csv: strain, Country, Collection.Year,
Source.Type, Host, Infection.status, published_classifier_prediction),
merges them on strain, and:

  - splits 80/20 (stratified on is_resistant)
  - trains Random Forest + XGBoost, cross-validated on the training fold
    (accuracy/precision/recall/F1/AUC) and evaluated for real on the held-out
    test set (same metrics)
  - benchmarks the paper's own published Kleborate classifier prediction
    (published_classifier_prediction) against the identical held-out test
    set and identical true labels -- the real comparison, not a sanity check
  - generates a SHAP summary plot (results/ml_classifier/shap_summary.png)
    for whichever of the two trained models scored higher test AUC
  - writes every metric (both models' CV + test results, the benchmark's
    results, top SHAP features) to results/ml_classifier/stage5_results.json

With --compare-splits instead, it runs a leakage check: the same features,
labels, and models, but three different train/test splits --
  - random: the 80/20 stratified split above
  - st_grouped: StratifiedGroupKFold grouped on ST (sequence type) -- no
    strain sharing a sequence type with a test-set strain appears in
    training. Near-identical clonal genomes landing on both sides of a
    random split would otherwise let a model do well by memorizing a clone
    rather than learning the resistance-determinant biology.
  - study_grouped: StratifiedGroupKFold grouped on Study.Accession -- no
    strain from a test-set strain's own source study appears in training,
    closer to the paper's own external-validation structure.
and prints/saves a single comparison table (test-set
accuracy/precision/recall/F1/AUC per model per split, plus each grouped
split's drop relative to the random-split baseline) to
results/ml_classifier/stage5_grouped_split_comparison.json.
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_validate, train_test_split
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FEATURES_CSV = PROJECT_ROOT / "data" / "processed" / "klebnet_features.csv"
DEFAULT_METADATA_CSV = PROJECT_ROOT / "data" / "processed" / "klebnet_metadata.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "ml_classifier"

STRAIN_COL = "strain"
LABEL_COL = "is_resistant"
BENCHMARK_COL = "published_classifier_prediction"
METADATA_COLS = ["Country", "Collection.Year", "Source.Type", "Host", "Infection.status", "ST", "Study.Accession"]
GROUP_COLS = {"st_grouped": "ST", "study_grouped": "Study.Accession"}

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_CV_SPLITS = 5
N_TOP_SHAP_FEATURES = 10


def load_data(features_csv, metadata_csv):
    """
    Load the feature matrix + metadata CSVs and merge them on strain.
    Returns (merged_df, feature_cols).
    """
    features = pd.read_csv(features_csv)
    metadata = pd.read_csv(metadata_csv)

    if STRAIN_COL not in features.columns:
        raise ValueError(f"Features CSV is missing required column '{STRAIN_COL}'")
    if LABEL_COL not in features.columns:
        raise ValueError(f"Features CSV is missing required column '{LABEL_COL}'")
    if STRAIN_COL not in metadata.columns:
        raise ValueError(f"Metadata CSV is missing required column '{STRAIN_COL}'")
    if BENCHMARK_COL not in metadata.columns:
        raise ValueError(f"Metadata CSV is missing required column '{BENCHMARK_COL}'")

    merged = features.merge(
        metadata[[STRAIN_COL, BENCHMARK_COL] + METADATA_COLS],
        on=STRAIN_COL, how="inner",
    )

    dropped_from_features = set(features[STRAIN_COL]) - set(merged[STRAIN_COL])
    dropped_from_metadata = set(metadata[STRAIN_COL]) - set(merged[STRAIN_COL])
    if dropped_from_features:
        warnings.warn(
            f"{len(dropped_from_features)} strain(s) in the features CSV had "
            f"no matching row in the metadata CSV and were dropped."
        )
    if dropped_from_metadata:
        warnings.warn(
            f"{len(dropped_from_metadata)} strain(s) in the metadata CSV had "
            f"no matching row in the features CSV (not part of this analysis)."
        )
    if merged.empty:
        raise ValueError(
            "No strains in common between the features and metadata CSVs -- "
            "check that 'strain' values match in both files."
        )

    feature_cols = [c for c in features.columns if c not in (STRAIN_COL, LABEL_COL)]
    return merged, feature_cols


def map_benchmark_prediction(series: pd.Series) -> np.ndarray:
    """
    Map the published classifier's categorical prediction (e.g. "nonwildtype
    R", "wildtype S", "nonwildtype I") to binary {0, 1}. A prediction that
    doesn't commit to R or S (e.g. "nonwildtype I") maps to -1: it can never
    equal a true label of 0 or 1, so it correctly counts as wrong for
    accuracy, and sklearn's precision/recall (pos_label=1) correctly exclude
    it from the positive-prediction count while still counting it as a
    missed positive for recall when the true label is 1. An abstention is
    not a correct prediction.
    """
    def _map(v):
        v = str(v)
        if v.endswith(" R"):
            return 1
        if v.endswith(" S"):
            return 0
        return -1
    return series.map(_map).to_numpy()


def _classification_metrics(y_true, y_pred, y_score=None) -> dict:
    """
    Compute accuracy/precision/recall(/F1/AUC) by hand rather than via
    sklearn's precision_score/recall_score: those reject y_pred containing
    any value outside {0, 1} (raise "Target is multiclass but
    average='binary'"), which breaks on the benchmark's -1 ("abstained")
    sentinel from map_benchmark_prediction. TP/FP/FN counted directly here
    handles that sentinel correctly (never a positive prediction, always a
    missed positive for recall when true label is 1) and gives identical
    results to sklearn's binary precision/recall for ordinary {0, 1}
    predictions (RF/XGBoost), so one implementation covers both cases.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    accuracy = float(np.mean(y_true == y_pred))
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred != 1) & (y_true == 1)))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    metrics = {"accuracy": accuracy, "precision": precision, "recall": recall}
    if y_score is not None:
        metrics["f1"] = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        metrics["auc"] = float(roc_auc_score(y_true, y_score))
    return {k: float(v) for k, v in metrics.items()}


def cross_validate_model(model, X_train, y_train) -> dict:
    cv = StratifiedKFold(n_splits=N_CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scoring = {"accuracy": "accuracy", "precision": "precision", "recall": "recall",
               "f1": "f1", "auc": "roc_auc"}
    scores = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring)
    return {metric: float(scores[f"test_{metric}"].mean()) for metric in scoring}


def evaluate_on_test(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1]
    return _classification_metrics(y_test, y_pred, y_score)


def make_grouped_split(merged: pd.DataFrame, group_col: str):
    """
    Split merged into train/test via StratifiedGroupKFold(n_splits=5) grouped
    on group_col, taking the first fold as the held-out test set (~20%,
    approximately -- exact size depends on how evenly group_col's group
    sizes divide, since no group may be split across train/test). Rows with
    a missing group_col value are dropped first (a NaN can't be assigned to
    a group).
    """
    df = merged.dropna(subset=[group_col]).copy()
    n_dropped = len(merged) - len(df)
    if n_dropped:
        print(f"  ({n_dropped} row(s) with missing '{group_col}' dropped -- can't be grouped)")

    sgkf = StratifiedGroupKFold(n_splits=N_CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    train_idx, test_idx = next(sgkf.split(df, df[LABEL_COL], groups=df[group_col]))
    train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]

    overlap = set(train_df[group_col]) & set(test_df[group_col])
    if overlap:
        raise AssertionError(
            f"Group leakage in '{group_col}' split: {len(overlap)} group(s) "
            f"appear in both train and test -- StratifiedGroupKFold should "
            f"never do this."
        )

    print(
        f"  train={len(train_df)} ({train_df[group_col].nunique()} unique "
        f"{group_col} groups), test={len(test_df)} "
        f"({test_df[group_col].nunique()} unique {group_col} groups), "
        f"0 groups shared (verified)"
    )
    return train_df, test_df


def fit_and_evaluate_models(X_train, y_train, X_test, y_test) -> dict:
    """Fit RF + XGBoost on (X_train, y_train), evaluate on the held-out
    test set only -- no training-fold CV. Used by the split-strategy
    comparison, where the metric that matters is held-out performance under
    each split strategy, not an additional CV robustness check."""
    models = {
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
        "xgboost": XGBClassifier(n_estimators=200, random_state=RANDOM_STATE, eval_metric="logloss"),
    }
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        results[name] = evaluate_on_test(model, X_test, y_test)
    return results


def compare_split_strategies(merged: pd.DataFrame, feature_cols: list, output_dir: Path) -> dict:
    """
    Run the random 80/20 split and the two group-aware splits (ST, Study.
    Accession), fit RF + XGBoost on each, and return {split_name: {model:
    metrics}}.
    """
    strategy_results = {}
    strategy_sizes = {}

    print("\n[random] 80/20 stratified split...")
    train_df, test_df = train_test_split(
        merged, test_size=TEST_SIZE, stratify=merged[LABEL_COL], random_state=RANDOM_STATE
    )
    print(f"  train={len(train_df)}, test={len(test_df)}")
    strategy_results["random"] = fit_and_evaluate_models(
        train_df[feature_cols], train_df[LABEL_COL], test_df[feature_cols], test_df[LABEL_COL]
    )
    strategy_sizes["random"] = {"n_train": len(train_df), "n_test": len(test_df)}

    for strategy_name, group_col in GROUP_COLS.items():
        print(f"\n[{strategy_name}] StratifiedGroupKFold grouped on '{group_col}'...")
        train_df, test_df = make_grouped_split(merged, group_col)
        strategy_results[strategy_name] = fit_and_evaluate_models(
            train_df[feature_cols], train_df[LABEL_COL], test_df[feature_cols], test_df[LABEL_COL]
        )
        strategy_sizes[strategy_name] = {"n_train": len(train_df), "n_test": len(test_df)}

    print_split_comparison_table(strategy_results)

    deltas = {
        strategy: {
            model: {
                metric: round(strategy_results[strategy][model][metric] - strategy_results["random"][model][metric], 4)
                for metric in strategy_results["random"][model]
            }
            for model in strategy_results["random"]
        }
        for strategy in strategy_results if strategy != "random"
    }

    summary = {
        "split_sizes": strategy_sizes,
        "metrics": strategy_results,
        "deltas_vs_random": deltas,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "stage5_grouped_split_comparison.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSplit-strategy comparison saved to: {out_path}")
    return summary


def print_split_comparison_table(strategy_results: dict):
    print(f"\n{'='*78}")
    print("SPLIT STRATEGY COMPARISON (random vs. clonal-/study-grouped, held-out test set)")
    print(f"{'='*78}")
    model_names = list(next(iter(strategy_results.values())).keys())
    for model_name in model_names:
        print(f"\n{model_name}")
        print(f"  {'Split':<16} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>8} {'AUC':>8}")
        print("  " + "-" * 68)
        random_m = strategy_results["random"][model_name]
        for strategy_name, per_model in strategy_results.items():
            m = per_model[model_name]
            print(
                f"  {strategy_name:<16} {m['accuracy']:>10.3f} {m['precision']:>10.3f} "
                f"{m['recall']:>10.3f} {m['f1']:>8.3f} {m['auc']:>8.3f}"
            )
        for strategy_name, per_model in strategy_results.items():
            if strategy_name == "random":
                continue
            m = per_model[model_name]
            d_acc = m["accuracy"] - random_m["accuracy"]
            d_auc = m["auc"] - random_m["auc"]
            print(f"  Δ {strategy_name:<14} accuracy {d_acc:+.3f}   AUC {d_auc:+.3f}")


def train_and_evaluate(X_train, y_train, X_test, y_test):
    models = {
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
        "xgboost": XGBClassifier(n_estimators=200, random_state=RANDOM_STATE, eval_metric="logloss"),
    }

    cv_results, test_results, fitted = {}, {}, {}
    for name, model in models.items():
        print(f"\n{name}: {N_CV_SPLITS}-fold stratified CV on training set...")
        cv_results[name] = cross_validate_model(model, X_train, y_train)
        for metric, value in cv_results[name].items():
            print(f"  cv {metric:10s}: {value:.3f}")

        model.fit(X_train, y_train)
        fitted[name] = model
        test_results[name] = evaluate_on_test(model, X_test, y_test)
        print(f"  held-out test set:")
        for metric, value in test_results[name].items():
            print(f"  test {metric:9s}: {value:.3f}")

    return cv_results, test_results, fitted


def benchmark_published_classifier(test_df: pd.DataFrame, y_test) -> dict:
    y_bench_pred = map_benchmark_prediction(test_df[BENCHMARK_COL])
    n_abstained = int((y_bench_pred == -1).sum())
    metrics = _classification_metrics(y_test, y_bench_pred)
    metrics["n_abstained"] = n_abstained
    if n_abstained:
        print(
            f"  ({n_abstained}/{len(y_bench_pred)} published predictions were "
            f"non-committal (\"nonwildtype I\") -- scored as wrong, not excluded)"
        )
    return metrics


def generate_shap_summary(model, X_test, output_dir: Path, model_name: str):
    """
    Generate a SHAP summary plot for the given fitted model on the held-out
    test set, showing which genetic features drive predictions, save it to
    output_dir/shap_summary.png, and return the top N features by mean
    absolute SHAP value.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Binary classification: some explainers/models return a per-class list
    # or a 3D array (samples, features, classes) -- normalize to the
    # positive class's contributions for a single, readable summary plot.
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
    top_features_list = [
        {"feature": feat, "mean_abs_shap": float(val)} for feat, val in top_features.items()
    ]

    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "shap_summary.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSHAP summary plot ({model_name}, on held-out test set) saved to: {out_path}")

    return out_path, top_features_list


def print_comparison_table(test_results: dict, benchmark_metrics: dict):
    print(f"\n{'='*70}")
    print("TEST-SET COMPARISON (held-out, 20% split)")
    print(f"{'='*70}")
    print(f"{'Model':<22} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>8} {'AUC':>8}")
    print("-" * 70)
    for name, m in test_results.items():
        print(
            f"{name:<22} {m['accuracy']:>10.3f} {m['precision']:>10.3f} "
            f"{m['recall']:>10.3f} {m['f1']:>8.3f} {m['auc']:>8.3f}"
        )
    print(
        f"{'published_classifier':<22} {benchmark_metrics['accuracy']:>10.3f} "
        f"{benchmark_metrics['precision']:>10.3f} {benchmark_metrics['recall']:>10.3f} "
        f"{'--':>8} {'--':>8}"
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Train Random Forest and XGBoost classifiers on the KlebNET-GSP "
            "feature matrix, benchmark against the paper's own published "
            "classifier prediction, with SHAP explainability."
        )
    )
    parser.add_argument("--features-csv", default=str(DEFAULT_FEATURES_CSV),
                         help="Feature matrix + label CSV from scripts/prep_klebnet_data.py")
    parser.add_argument("--metadata-csv", default=str(DEFAULT_METADATA_CSV),
                         help="Metadata + benchmark prediction CSV from scripts/prep_klebnet_data.py")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR),
                         help="Directory to write results (SHAP plot, results JSON) to")
    parser.add_argument(
        "--compare-splits", action="store_true",
        help=(
            "Instead of the normal single-split pipeline, run a leakage "
            "check: random split vs. StratifiedGroupKFold grouped on ST vs. "
            "grouped on Study.Accession, RF + XGBoost on each, one "
            "comparison table. See module docstring."
        ),
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    try:
        merged, feature_cols = load_data(args.features_csv, args.metadata_csv)
        print(f"Loaded {merged.shape[0]} strains x {len(feature_cols)} features")

        if args.compare_splits:
            compare_split_strategies(merged, feature_cols, output_dir)
            return

        train_df, test_df = train_test_split(
            merged, test_size=TEST_SIZE, stratify=merged[LABEL_COL], random_state=RANDOM_STATE
        )
        X_train, y_train = train_df[feature_cols], train_df[LABEL_COL]
        X_test, y_test = test_df[feature_cols], test_df[LABEL_COL]
        print(
            f"Train/test split: {len(train_df)}/{len(test_df)} "
            f"({(1-TEST_SIZE):.0%}/{TEST_SIZE:.0%}, stratified on {LABEL_COL})"
        )

        cv_results, test_results, fitted = train_and_evaluate(X_train, y_train, X_test, y_test)

        print(f"\nBenchmarking published_classifier_prediction on the held-out test set...")
        benchmark_metrics = benchmark_published_classifier(test_df, y_test)

        print_comparison_table(test_results, benchmark_metrics)

        # SHAP on whichever model actually scored higher on the held-out
        # test set, not an arbitrary default.
        shap_model_name = max(test_results, key=lambda name: test_results[name]["auc"])
        shap_path, top_shap_features = generate_shap_summary(
            fitted[shap_model_name], X_test, output_dir, shap_model_name
        )

        print(f"\nTop {N_TOP_SHAP_FEATURES} SHAP features ({shap_model_name}):")
        for i, item in enumerate(top_shap_features, 1):
            print(f"  {i:2d}. {item['feature']:<20} {item['mean_abs_shap']:.4f}")

        results_summary = {
            "n_train": len(train_df),
            "n_test": len(test_df),
            "n_features": len(feature_cols),
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "cv_metrics": cv_results,
            "test_metrics": test_results,
            "benchmark_metrics": {"published_classifier": benchmark_metrics},
            "shap_model_used": shap_model_name,
            "shap_top_features": top_shap_features,
            "shap_summary_plot": str(shap_path),
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        results_path = output_dir / "stage5_results.json"
        with open(results_path, "w") as f:
            json.dump(results_summary, f, indent=2)
        print(f"\nResults summary saved to: {results_path}")

    except Exception as e:
        sys.exit(f"ERROR: {e}")


if __name__ == "__main__":
    main()

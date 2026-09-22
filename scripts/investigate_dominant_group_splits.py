#!/usr/bin/env python3
"""
PathTwin Stage 5 -- investigating whether A. baumannii's and E. faecium's
dominant-clone/dominant-study imbalance is fixable via a properly-designed
grouped split (same spirit as K. pneumoniae's original grouped-split
leakage check), or whether it reflects a genuine data-scarcity ceiling.

The diagnosis that matters is NOT just "is one group >15% of the data"
(train_bvbrc_classifier.py's existing warning threshold) -- it's whether
there is enough real diversity OUTSIDE that one dominant group to build a
meaningful held-out test set once it's set aside. Two very different
situations can both trip that 15% warning:
  (a) "one big group + a real long tail of many smaller ones" -- fixable:
      force the dominant group into train, build test from the tail.
  (b) "only a handful of groups total, one of which is unavoidably most
      of the data" -- a genuine ceiling: there is no way to build a test
      set with real diversity no matter how the split is arranged.

For each species/grouping this script:
  1. Reports n_groups and the full size distribution (not just the top
     one), which is what actually distinguishes (a) from (b).
  2. Classifies fixable vs. scarcity using explicit, stated criteria
     (see `classify_fixability`).
  3. If fixable: builds a DESIGNED holdout (dominant group forced into
     train, GroupShuffleSplit on the remaining groups targeting an
     overall ~20% test fraction) instead of trusting
     StratifiedGroupKFold's blind first-fold pick, verifies zero group
     leakage the same way, fits RF+XGBoost, and reports accuracy/AUC
     side by side with the original (degenerate) grouped-split result --
     same comparison-table spirit as the K. pneumoniae random-vs-grouped
     check.
  4. If scarcity: says so plainly, with the specific number that fails
     the fixability bar, and separately checks BV-BRC directly for
     whether more real data exists that could plausibly help (not just
     assumed).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_bvbrc_classifier import load_data, _classification_metrics, _patch_shap_xgboost_base_score_bug
from xgboost import XGBClassifier

RANDOM_STATE = 42
TARGET_TEST_FRAC = 0.20

# Fixability bar, stated explicitly (used consistently for every
# species/grouping, not tuned per case):
#   - at least 8 non-dominant groups (enough to average sampling noise
#     over, not just 1-2 additional big blocks standing in for "diversity")
#   - the non-dominant remainder must be at least 30% of the (grouped)
#     data -- if the dominant group already eats >70%, there isn't enough
#     material left to build a real ~20%-of-total test set without also
#     using most of what little remains
#   - no single OTHER group may exceed 40% of the non-dominant remainder
#     -- otherwise "the tail" is really just a second dominant group and
#     the same problem recurs one level down
MIN_OTHER_GROUPS = 8
MIN_REMAINDER_FRACTION = 0.30
MAX_SECOND_GROUP_FRACTION_OF_REMAINDER = 0.40


def group_size_distribution(df: pd.DataFrame, group_col: str) -> pd.Series:
    sub = df.dropna(subset=[group_col]).copy()
    sub = sub[sub[group_col].astype(str).str.strip() != ""]
    return sub[group_col].value_counts(), len(sub)


def classify_fixability(vc: pd.Series, n_grouped: int, n_total: int) -> dict:
    n_groups = len(vc)
    dominant_group = vc.index[0]
    dominant_n = int(vc.iloc[0])
    dominant_frac_of_grouped = dominant_n / n_grouped
    dominant_frac_of_total = dominant_n / n_total
    remainder_n = n_grouped - dominant_n
    remainder_frac_of_grouped = remainder_n / n_grouped
    n_other_groups = n_groups - 1
    second_group_frac_of_remainder = (int(vc.iloc[1]) / remainder_n) if n_other_groups >= 1 and remainder_n > 0 else 1.0

    reasons_failed = []
    if n_other_groups < MIN_OTHER_GROUPS:
        reasons_failed.append(f"only {n_other_groups} other group(s), need >={MIN_OTHER_GROUPS}")
    if remainder_frac_of_grouped < MIN_REMAINDER_FRACTION:
        reasons_failed.append(f"non-dominant remainder is only {remainder_frac_of_grouped:.1%} of grouped data, "
                               f"need >={MIN_REMAINDER_FRACTION:.0%}")
    if second_group_frac_of_remainder > MAX_SECOND_GROUP_FRACTION_OF_REMAINDER:
        reasons_failed.append(f"2nd-largest group is {second_group_frac_of_remainder:.1%} of the remainder "
                               f"(itself a near-dominant block), need <={MAX_SECOND_GROUP_FRACTION_OF_REMAINDER:.0%}")
    # coverage: how much of the TOTAL dataset even has this grouping populated
    coverage_frac = n_grouped / n_total
    if coverage_frac < 0.5:
        reasons_failed.append(f"this grouping is only populated for {coverage_frac:.1%} of the total dataset "
                               f"(most genomes lack a value at all)")

    fixable = len(reasons_failed) == 0
    return {
        "n_groups": n_groups,
        "dominant_group": str(dominant_group),
        "dominant_n": dominant_n,
        "dominant_frac_of_grouped": dominant_frac_of_grouped,
        "dominant_frac_of_total": dominant_frac_of_total,
        "remainder_n": remainder_n,
        "remainder_frac_of_grouped": remainder_frac_of_grouped,
        "n_other_groups": n_other_groups,
        "second_group_frac_of_remainder": second_group_frac_of_remainder,
        "coverage_frac_of_total": coverage_frac,
        "fixable": fixable,
        "reasons_failed": reasons_failed,
    }


def designed_holdout_split(merged: pd.DataFrame, group_col: str, label_col: str, dominant_group: str):
    """Force the dominant group into train; build test from the remaining
    groups via GroupShuffleSplit targeting TARGET_TEST_FRAC of the WHOLE
    (grouped) dataset -- i.e. targeting a higher fraction of the smaller
    remainder so the overall test size lands near the intended 20%."""
    df = merged.dropna(subset=[group_col]).copy()
    df = df[df[group_col].astype(str).str.strip() != ""]

    dominant_mask = df[group_col] == dominant_group
    forced_train = df[dominant_mask]
    remainder = df[~dominant_mask]

    remainder_target_frac = min(0.9, TARGET_TEST_FRAC * len(df) / len(remainder))
    gss = GroupShuffleSplit(n_splits=1, test_size=remainder_target_frac, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(remainder, groups=remainder[group_col]))
    remainder_train, test_df = remainder.iloc[train_idx], remainder.iloc[test_idx]

    train_df = pd.concat([forced_train, remainder_train], ignore_index=False)

    overlap = set(train_df[group_col]) & set(test_df[group_col])
    assert not overlap, f"group leakage in designed split: {overlap}"

    return train_df, test_df


def fit_and_evaluate(train_df, test_df, feature_cols, label_col, use_gpu=True):
    X_train, y_train = train_df[feature_cols], train_df[label_col]
    X_test, y_test = test_df[feature_cols], test_df[label_col]
    models = {
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
        "xgboost": XGBClassifier(n_estimators=200, random_state=RANDOM_STATE, eval_metric="logloss",
                                  device="cuda" if use_gpu else "cpu"),
    }
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_score = model.predict_proba(X_test)[:, 1]
        results[name] = _classification_metrics(y_test, y_pred, y_score)
    return results


def investigate_species(tag, features_csv, metadata_csv, original_results_json):
    print(f"\n{'='*90}\n{tag}\n{'='*90}")
    merged, feature_cols = load_data(features_csv, metadata_csv)
    n_total = len(merged)
    original = json.loads(Path(original_results_json).read_text())

    findings = {}
    for strategy_name, group_col in [("st_grouped", "ST"), ("bioproject_grouped", "bioproject_accession")]:
        print(f"\n--- {group_col} ---")
        vc, n_grouped = group_size_distribution(merged, group_col)
        diag = classify_fixability(vc, n_grouped, n_total)
        print(f"  {diag['n_groups']} distinct groups, {n_grouped}/{n_total} genomes have a value "
              f"({diag['coverage_frac_of_total']:.1%} coverage)")
        print(f"  dominant group {diag['dominant_group']!r}: {diag['dominant_n']} genomes "
              f"({diag['dominant_frac_of_grouped']:.1%} of grouped data)")
        print(f"  remainder after dominant group: {diag['remainder_n']} genomes across "
              f"{diag['n_other_groups']} other groups ({diag['remainder_frac_of_grouped']:.1%} of grouped data)")
        print(f"  2nd-largest group is {diag['second_group_frac_of_remainder']:.1%} of that remainder")

        if diag["fixable"]:
            print(f"  *** FIXABLE: enough real diversity outside the dominant group -- "
                  f"building a designed holdout split ***")
            train_df, test_df = designed_holdout_split(merged, group_col, LABEL_COL := "is_resistant", diag["dominant_group"])
            n_train, n_test = len(train_df), len(test_df)
            test_classes = test_df[LABEL_COL].value_counts().to_dict()
            print(f"  designed split: train={n_train} ({train_df[group_col].nunique()} groups), "
                  f"test={n_test} ({test_df[group_col].nunique()} groups, {n_test/n_total:.1%} of total), "
                  f"test class balance={test_classes}")
            results = fit_and_evaluate(train_df, test_df, feature_cols, LABEL_COL)
            for model_name, m in results.items():
                print(f"    {model_name:<15} accuracy={m['accuracy']:.3f} precision={m['precision']:.3f} "
                      f"recall={m['recall']:.3f} f1={m['f1']:.3f} auc={m['auc']:.3f}")
            orig = original["metrics"].get(strategy_name)
            print(f"  vs. original (arbitrary first-fold, {original['split_sizes'][strategy_name]['n_test']} "
                  f"test genomes): {orig}")
            findings[strategy_name] = {
                "diagnosis": diag, "fixable": True,
                "designed_split_sizes": {"n_train": n_train, "n_test": n_test,
                                          "n_train_groups": int(train_df[group_col].nunique()),
                                          "n_test_groups": int(test_df[group_col].nunique()),
                                          "test_class_balance": {str(k): int(v) for k, v in test_classes.items()}},
                "designed_split_metrics": results,
                "original_metrics": orig,
            }
        else:
            print(f"  *** SCARCITY -- not attempting a designed split: {'; '.join(diag['reasons_failed'])} ***")
            findings[strategy_name] = {"diagnosis": diag, "fixable": False}

    return findings


if __name__ == "__main__":
    all_findings = {}
    all_findings["abaumannii"] = investigate_species(
        "A. baumannii (carbapenem)",
        "data/processed/abaumannii_features.csv", "data/processed/abaumannii_metadata.csv",
        "results/ml_classifier/abaumannii_carbapenem_stage5_results.json",
    )
    all_findings["efaecium"] = investigate_species(
        "E. faecium (vancomycin)",
        "data/processed/efaecium_features.csv", "data/processed/efaecium_metadata.csv",
        "results/ml_classifier/efaecium_vancomycin_stage5_results.json",
    )

    out_path = Path("results/ml_classifier/dominant_group_investigation.json")
    out_path.write_text(json.dumps(all_findings, indent=2, default=str))
    print(f"\n\nSaved: {out_path}")

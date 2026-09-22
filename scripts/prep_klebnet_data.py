#!/usr/bin/env python3
"""
PathTwin Stage 5 — KlebNET-GSP training data preparation

Loads the KlebNET-GSP supplementary genotype/phenotype tables (TableS2:
per-strain Kleborate-style genotype calls; TableS3: per-strain AST
phenotypes + collection metadata), filters to ciprofloxacin, merges them on
strain, and builds:

  - data/processed/klebnet_features.csv  -- feature matrix + label, one row
    per strain, for training a classifier
  - data/processed/klebnet_metadata.csv  -- strain + collection metadata
    (including ST and Study.Accession, for clonal- and study-leakage-aware
    grouped train/test splits) + the paper's own published Kleborate
    classifier prediction, kept entirely separate from the features/label
    above

Feature selection is deliberately narrow and mechanistic: only the
individual GyrA-/ParC- QRDR allele indicator columns, the individual qnr/
Qnr PMQR allele columns, qepA2, the aac(6')-Ib-cr variant-call columns, and
oqxA_copy/oqxB_copy. Everything else in TableS2 -- including the summary
columns that share a name prefix with those (GyrA-83, GyrA-87, ParC-80,
ParC-84, GyrA_mutations, ParC_mutations, qnr, qep, aac6, aac6_allele) and
every QRDR_mutations/resistant/WT_vs_nonWT/nonWT_binary/Ciprofloxacin_*
column -- is excluded on principle: those are the paper's own derived
summaries or its classifier's own predictions, and training on them would
leak the answer (directly, in the case of the Ciprofloxacin_* columns, or
via near-duplicate redundancy with the individual allele columns, in the
case of the summary columns).
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GENOTYPES_TSV = PROJECT_ROOT / "data" / "raw" / "klebnet_gsp" / "TableS2_genotypes.tsv"
DEFAULT_PHENOTYPES_TSV = PROJECT_ROOT / "data" / "raw" / "klebnet_gsp" / "TableS3_phenotypes_metadata.tsv"
DEFAULT_FEATURES_OUT = PROJECT_ROOT / "data" / "processed" / "klebnet_features.csv"
DEFAULT_METADATA_OUT = PROJECT_ROOT / "data" / "processed" / "klebnet_metadata.csv"

ANTIBIOTIC = "ciprofloxacin"
STRAIN_COL = "strain"
PHENOTYPE_COL = "Resistance.phenotype"
LABEL_COL = "is_resistant"
BENCHMARK_SRC_COL = "Ciprofloxacin_prediction"
BENCHMARK_OUT_COL = "published_classifier_prediction"
METADATA_COLS = [
    "Country", "Collection.Year", "Source.Type", "Host", "Infection.status",
    "ST",               # TableS2 (genotypes) -- sequence type, for clonal-leakage-aware grouped splits
    "Study.Accession",  # TableS3 (phenotypes) -- for study/batch-leakage-aware grouped splits
]

# Individual allele/variant-call columns only -- see module docstring for
# exactly what this excludes and why.
_FEATURE_REGEXES = [
    re.compile(r"^GyrA-\d+[A-Za-z]$"),   # GyrA-83F, GyrA-87Y, ... (not the bare GyrA-83/GyrA-87 summaries)
    re.compile(r"^ParC-\d+[A-Za-z]$"),   # ParC-80I, ParC-84V, ... (not the bare ParC-80/ParC-84 summaries)
    re.compile(r"^[Qq]nr[A-Z]"),         # qnrA1, qnrB17, QnrB76, qnrS1*, ... (not the bare "qnr" summary)
]
_FEATURE_LITERAL_COLS = ["qepA2", "oqxA_copy", "oqxB_copy"]
_AAC6_IBCR_PREFIX = "aac(6')-Ib-cr"      # aac(6')-Ib-cr.v1/.v2/.v2?/.v2*


def resolve_feature_columns(genotypes: pd.DataFrame) -> list:
    """Resolve the exact TableS2 columns that qualify as model features."""
    cols = [
        c for c in genotypes.columns
        if any(p.match(c) for p in _FEATURE_REGEXES)
        or c in _FEATURE_LITERAL_COLS
        or c.startswith(_AAC6_IBCR_PREFIX)
    ]
    if not cols:
        raise ValueError(
            "No feature columns matched in TableS2 -- column naming may have "
            "changed. Expected GyrA-/ParC- allele columns, qnr/Qnr allele "
            "columns, qepA2, aac(6')-Ib-cr.* columns, oqxA_copy, oqxB_copy."
        )
    return cols


def load_tables(genotypes_tsv, phenotypes_tsv):
    genotypes = pd.read_csv(genotypes_tsv, sep="\t", low_memory=False)
    phenotypes = pd.read_csv(phenotypes_tsv, sep="\t", low_memory=False)
    for name, df in (("TableS2 genotypes", genotypes), ("TableS3 phenotypes", phenotypes)):
        if STRAIN_COL not in df.columns:
            raise ValueError(f"{name} table is missing the '{STRAIN_COL}' column")
    return genotypes, phenotypes


def filter_to_antibiotic(phenotypes: pd.DataFrame, antibiotic: str = ANTIBIOTIC) -> pd.DataFrame:
    total = len(phenotypes)
    filtered = phenotypes[phenotypes["Antibiotic"] == antibiotic].copy()
    if len(filtered) == total:
        print(f"Antibiotic filter: all {total} TableS3 rows are '{antibiotic}' (confirmed).")
    else:
        other = sorted(set(phenotypes["Antibiotic"]) - {antibiotic})
        print(
            f"Antibiotic filter: {len(filtered)}/{total} TableS3 rows are "
            f"'{antibiotic}' -- {total - len(filtered)} row(s) dropped "
            f"(other antibiotics present: {other})."
        )
    return filtered


def merge_tables(genotypes: pd.DataFrame, phenotypes_cipro: pd.DataFrame) -> pd.DataFrame:
    merged = phenotypes_cipro.merge(genotypes, on=STRAIN_COL, how="inner", validate="one_to_one")
    unmatched_pheno = set(phenotypes_cipro[STRAIN_COL]) - set(genotypes[STRAIN_COL])
    unmatched_geno = set(genotypes[STRAIN_COL]) - set(phenotypes_cipro[STRAIN_COL])
    if unmatched_pheno:
        print(
            f"WARNING: {len(unmatched_pheno)} strain(s) in TableS3 (ciprofloxacin) "
            f"had no match in TableS2 and were dropped by the merge."
        )
    if unmatched_geno:
        print(
            f"Note: {len(unmatched_geno)} strain(s) in TableS2 had no ciprofloxacin "
            f"phenotype in TableS3 and are not part of this analysis."
        )
    return merged


def build_label(merged: pd.DataFrame) -> pd.DataFrame:
    """
    Add the binary is_resistant label from Resistance.phenotype (R->1, S->0),
    dropping "I" (intermediate) rows for the primary model, matching the
    paper's own R-vs-S approach.
    """
    valid_phenotypes = {"R", "S"}
    before = len(merged)
    dropped = merged[~merged[PHENOTYPE_COL].isin(valid_phenotypes)]
    kept = merged[merged[PHENOTYPE_COL].isin(valid_phenotypes)].copy()

    if len(dropped):
        drop_counts = dropped[PHENOTYPE_COL].value_counts(dropna=False).to_dict()
        print(
            f"Dropping {len(dropped)}/{before} row(s) with non-R/S phenotype "
            f"for the primary model ({drop_counts})."
        )
    else:
        print(f"No non-R/S rows to drop ({before} rows all R or S).")

    kept[LABEL_COL] = (kept[PHENOTYPE_COL] == "R").astype(int)
    return kept


def build_outputs(analysis_df: pd.DataFrame, feature_cols: list):
    features_df = analysis_df[[STRAIN_COL, LABEL_COL] + feature_cols].copy()

    metadata_df = analysis_df[[STRAIN_COL] + METADATA_COLS + [BENCHMARK_SRC_COL]].copy()
    metadata_df = metadata_df.rename(columns={BENCHMARK_SRC_COL: BENCHMARK_OUT_COL})

    return features_df, metadata_df


def report_summary(features_df: pd.DataFrame, metadata_df: pd.DataFrame, feature_cols: list):
    n = len(features_df)
    class_counts = features_df[LABEL_COL].value_counts().rename({1: "R (resistant)", 0: "S (susceptible)"})
    n_countries = metadata_df["Country"].nunique()

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total isolates (after filtering + merge, R/S only): {n}")
    print(f"Class balance:")
    for label, count in class_counts.items():
        print(f"  {label:<16}: {count} ({count/n:.1%})")
    print(f"Number of features: {len(feature_cols)}")
    print(f"Countries represented: {n_countries}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare KlebNET-GSP ciprofloxacin genotype/phenotype data for training."
    )
    parser.add_argument("--genotypes-tsv", default=str(DEFAULT_GENOTYPES_TSV))
    parser.add_argument("--phenotypes-tsv", default=str(DEFAULT_PHENOTYPES_TSV))
    parser.add_argument("--features-out", default=str(DEFAULT_FEATURES_OUT))
    parser.add_argument("--metadata-out", default=str(DEFAULT_METADATA_OUT))
    args = parser.parse_args()

    try:
        genotypes, phenotypes = load_tables(args.genotypes_tsv, args.phenotypes_tsv)
        phenotypes_cipro = filter_to_antibiotic(phenotypes)
        merged = merge_tables(genotypes, phenotypes_cipro)
        analysis_df = build_label(merged)

        feature_cols = resolve_feature_columns(genotypes)
        features_df, metadata_df = build_outputs(analysis_df, feature_cols)

        features_out = Path(args.features_out)
        metadata_out = Path(args.metadata_out)
        features_out.parent.mkdir(parents=True, exist_ok=True)
        metadata_out.parent.mkdir(parents=True, exist_ok=True)
        features_df.to_csv(features_out, index=False)
        metadata_df.to_csv(metadata_out, index=False)

        print(f"\nFeatures + label -> {features_out} ({features_df.shape[0]} rows x {features_df.shape[1]} cols)")
        print(f"Metadata + benchmark prediction -> {metadata_out} ({metadata_df.shape[0]} rows x {metadata_df.shape[1]} cols)")

        report_summary(features_df, metadata_df, feature_cols)
    except Exception as e:
        sys.exit(f"ERROR: {e}")


if __name__ == "__main__":
    main()

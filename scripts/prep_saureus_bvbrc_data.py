#!/usr/bin/env python3
"""
PathTwin Stage 5 extension -- S. aureus AMR phenotype data prep (BV-BRC)

Pulls S. aureus oxacillin (methicillin-class) resistance phenotype data and
gene-presence features directly from BV-BRC's REST Data API
(https://www.bv-brc.org/api/) via plain HTTPS/JSON requests -- no
`p3-scripts` CLI needed for this, so no new conda env either (the task's
own "if needed" qualifier: it wasn't needed).

Labels: `genome_amr` collection, taxon_id=1280 (S. aureus), antibiotic=
oxacillin, `evidence="Laboratory Method"` only. BV-BRC's own ML-predicted
phenotypes (`evidence="Computational Method"`, e.g. "AdaBoost Classifier",
"MIC XGBoost Model") are excluded entirely -- same leakage-avoidance rule
as KlebNET-GSP's published_classifier_prediction column: never train on
another tool's prediction, only on real measured phenotypes.

Features: `sp_gene` collection, `property="Antibiotic Resistance"` -- raw
BLAT-based gene/protein hits against CARD/NDARO/PATRIC's own curated AMR
gene set (the same CARD this whole project already uses elsewhere), not a
resistance PREDICTION. Feature key is the gene symbol when present, else a
cleaned product name -- BV-BRC's CARD-sourced sp_gene rows sometimes omit
`gene` entirely even for well-known genes like mecA (both a `gene="mecA"`
row and a separate `gene=None`/"...MecA..." product row can exist for the
same genome; both become features here rather than silently dropping one).

Grouping metadata: the `genome` collection's own `mlst` and
`bioproject_accession` fields -- used for the grouped train/test split so
no genome sharing a clone (ST) or source study with a test-set genome
appears in training.
"""

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "bvbrc_saureus"
DEFAULT_FEATURES_OUT = PROJECT_ROOT / "data" / "processed" / "saureus_features.csv"
DEFAULT_METADATA_OUT = PROJECT_ROOT / "data" / "processed" / "saureus_metadata.csv"

API_BASE = "https://www.bv-brc.org/api"
TAXON_ID = 1280  # Staphylococcus aureus
ANTIBIOTIC = "oxacillin"
LABEL_COL = "is_resistant"
BATCH_SIZE = 150
SP_GENE_ROW_LIMIT = 25000  # generously above any plausible per-batch row count; checked against actual count below
REQUEST_TIMEOUT = 90
HEADERS = {"Accept": "application/json"}


def _rql_get(path: str, query: str, retries: int = 3) -> list:
    url = f"{API_BASE}/{path}/?{query}"
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, json.JSONDecodeError) as e:
            print(f"    (attempt {attempt + 1}/{retries} failed: {e}; retrying)", flush=True)
            if attempt == retries - 1:
                raise RuntimeError(f"BV-BRC API request failed after {retries} attempts: {url}\n{e}")
            time.sleep(2 * (attempt + 1))
    return []


def fetch_oxacillin_labels() -> list:
    """All S. aureus oxacillin AMR records with real (lab-measured) phenotypes."""
    query = (
        f"and(eq(taxon_id,{TAXON_ID}),eq(antibiotic,{ANTIBIOTIC}),"
        f'eq(evidence,"Laboratory Method"))&limit(5000,0)'
    )
    return _rql_get("genome_amr", query)


def resolve_labels(records: list) -> dict:
    """
    Collapse (possibly multiple) oxacillin records per genome into a single
    label. Drops blank/Intermediate phenotypes. A genome with conflicting
    Resistant/Susceptible calls across records is dropped entirely rather
    than guessed at -- real data quality issue (duplicate submissions,
    different studies), not resolved by majority vote.
    """
    by_genome = {}
    for r in records:
        pheno = r.get("resistant_phenotype", "")
        if pheno not in ("Resistant", "Susceptible"):
            continue
        by_genome.setdefault(r["genome_id"], set()).add(pheno)

    labels, dropped_conflicting = {}, []
    for genome_id, phenos in by_genome.items():
        if len(phenos) > 1:
            dropped_conflicting.append(genome_id)
            continue
        labels[genome_id] = 1 if phenos == {"Resistant"} else 0

    print(f"Resolved {len(labels)} genomes with a single, unambiguous oxacillin phenotype.")
    if dropped_conflicting:
        print(f"  Dropped {len(dropped_conflicting)} genome(s) with conflicting R/S calls across records.")
    return labels


def _batched(items, size):
    items = list(items)
    for i in range(0, len(items), size):
        yield items[i : i + size]


def fetch_genome_metadata(genome_ids: list) -> dict:
    """mlst + bioproject_accession per genome, for the grouped split."""
    metadata = {}
    batches = list(_batched(genome_ids, BATCH_SIZE))
    for i, batch in enumerate(batches, 1):
        print(f"  genome metadata batch {i}/{len(batches)}...", flush=True)
        ids = ",".join(batch)
        query = f"and(in(genome_id,({ids})))&select(genome_id,mlst,bioproject_accession)&limit({BATCH_SIZE},0)"
        for row in _rql_get("genome", query):
            metadata[row["genome_id"]] = {
                "mlst": row.get("mlst", ""),
                "bioproject_accession": row.get("bioproject_accession", ""),
            }
    print(f"Fetched metadata for {len(metadata)}/{len(genome_ids)} genomes.")
    return metadata


def _feature_key(row: dict) -> str:
    gene = (row.get("gene") or "").strip()
    if gene:
        return gene
    product = (row.get("product") or "").strip()
    return product or f"unnamed:{row.get('source_id', row.get('id', 'unknown'))}"


def fetch_sp_gene_features(genome_ids: list, checkpoint_path: Path = None) -> dict:
    """{genome_id: {feature_name: 1}} from sp_gene, property=Antibiotic Resistance.

    Writes a checkpoint after every batch (if checkpoint_path given) so a
    killed/interrupted run doesn't lose completed batches.
    """
    features = {gid: {} for gid in genome_ids}
    batches = list(_batched(genome_ids, BATCH_SIZE))
    for i, batch in enumerate(batches, 1):
        print(f"  sp_gene batch {i}/{len(batches)} ({len(batch)} genomes)...", flush=True)
        ids = ",".join(batch)
        query = (
            f'and(in(genome_id,({ids})),eq(property,"Antibiotic Resistance"))'
            f"&select(genome_id,gene,product,source_id,id)&limit({SP_GENE_ROW_LIMIT},0)"
        )
        rows = _rql_get("sp_gene", query)
        print(f"    -> {len(rows)} sp_gene rows", flush=True)
        if len(rows) >= SP_GENE_ROW_LIMIT:
            raise RuntimeError(
                f"sp_gene batch hit the row limit ({SP_GENE_ROW_LIMIT}) -- results are "
                f"likely truncated. Raise SP_GENE_ROW_LIMIT and/or shrink BATCH_SIZE."
            )
        for row in rows:
            gid = row["genome_id"]
            features.setdefault(gid, {})[_feature_key(row)] = 1
        if checkpoint_path:
            checkpoint_path.write_text(json.dumps(features))
    n_with_features = sum(1 for v in features.values() if v)
    print(f"Fetched sp_gene AMR features for {n_with_features}/{len(genome_ids)} genomes (rest had none).")
    return features


def build_outputs(labels: dict, metadata: dict, features: dict):
    genome_ids = sorted(labels.keys())

    feature_names = sorted({name for gid in genome_ids for name in features.get(gid, {})})
    feat_rows = []
    for gid in genome_ids:
        row = {"strain": gid, LABEL_COL: labels[gid]}
        gid_features = features.get(gid, {})
        for name in feature_names:
            row[name] = gid_features.get(name, 0)
        feat_rows.append(row)
    features_df = pd.DataFrame(feat_rows)

    meta_rows = []
    for gid in genome_ids:
        m = metadata.get(gid, {})
        meta_rows.append({"strain": gid, "ST": m.get("mlst", ""), "bioproject_accession": m.get("bioproject_accession", "")})
    metadata_df = pd.DataFrame(meta_rows)

    return features_df, metadata_df, feature_names


def main():
    parser = argparse.ArgumentParser(
        description="Prepare S. aureus oxacillin resistance genotype/phenotype data from BV-BRC."
    )
    parser.add_argument("--features-out", default=str(DEFAULT_FEATURES_OUT))
    parser.add_argument("--metadata-out", default=str(DEFAULT_METADATA_OUT))
    parser.add_argument("--cache-dir", default=str(RAW_DIR),
                         help="Directory to cache raw BV-BRC API responses (for reproducibility/re-runs)")
    args = parser.parse_args()

    try:
        cache_dir = Path(args.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        print(f"Fetching S. aureus {ANTIBIOTIC} AMR records (Laboratory Method only) from BV-BRC...")
        raw_records = fetch_oxacillin_labels()
        print(f"  {len(raw_records)} raw records fetched "
              f"(before dropping blank/Intermediate/conflicting phenotypes).")
        (cache_dir / "oxacillin_amr_lab.json").write_text(json.dumps(raw_records, indent=2))

        pheno_counts = Counter(r.get("resistant_phenotype", "(blank)") for r in raw_records)
        print(f"  Raw phenotype distribution: {dict(pheno_counts)}")

        labels = resolve_labels(raw_records)
        genome_ids = sorted(labels.keys())

        metadata_cache = cache_dir / "genome_metadata.json"
        if metadata_cache.exists():
            print(f"\nReusing cached genome metadata ({metadata_cache})...")
            metadata = json.loads(metadata_cache.read_text())
        else:
            print("\nFetching genome metadata (ST, BioProject) for grouped-split support...")
            metadata = fetch_genome_metadata(genome_ids)
            metadata_cache.write_text(json.dumps(metadata, indent=2))

        features_cache = cache_dir / "sp_gene_features.json"
        if features_cache.exists():
            print(f"\nReusing cached sp_gene features ({features_cache})...")
            features = json.loads(features_cache.read_text())
        else:
            print("\nFetching sp_gene antibiotic-resistance features...")
            checkpoint = cache_dir / "sp_gene_features.checkpoint.json"
            features = fetch_sp_gene_features(genome_ids, checkpoint_path=checkpoint)
            features_cache.write_text(json.dumps(features, indent=2))

        features_df, metadata_df, feature_names = build_outputs(labels, metadata, features)

        features_out = Path(args.features_out)
        metadata_out = Path(args.metadata_out)
        features_out.parent.mkdir(parents=True, exist_ok=True)
        metadata_out.parent.mkdir(parents=True, exist_ok=True)
        features_df.to_csv(features_out, index=False)
        metadata_df.to_csv(metadata_out, index=False)

        n = len(features_df)
        n_resistant = int(features_df[LABEL_COL].sum())
        n_with_st = (metadata_df["ST"] != "").sum()
        n_with_bioproject = (metadata_df["bioproject_accession"] != "").sum()

        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"Total genomes (unambiguous oxacillin phenotype): {n}")
        print(f"Class balance: {n_resistant} Resistant ({n_resistant/n:.1%}) / "
              f"{n - n_resistant} Susceptible ({(n - n_resistant)/n:.1%})")
        print(f"Number of features: {len(feature_names)}")
        print(f"Genomes with an ST (mlst) call: {n_with_st}/{n}")
        print(f"Genomes with a BioProject accession: {n_with_bioproject}/{n}")
        print(f"\nFeatures -> {features_out} ({features_df.shape[0]} rows x {features_df.shape[1]} cols)")
        print(f"Metadata -> {metadata_out} ({metadata_df.shape[0]} rows x {metadata_df.shape[1]} cols)")

    except Exception as e:
        sys.exit(f"ERROR: {e}")


if __name__ == "__main__":
    main()

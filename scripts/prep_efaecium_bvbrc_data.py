#!/usr/bin/env python3
"""
PathTwin Stage 5 extension -- E. faecium vancomycin resistance data prep
(BV-BRC). Single antibiotic (vancomycin, the defining VRE phenotype), so
this follows the S. aureus/oxacillin single-drug template directly rather
than the multi-drug-class pattern used for A. baumannii/P. aeruginosa.

Labels: `genome_amr` collection, taxon_id=1352 (E. faecium), antibiotic=
vancomycin, `evidence="Laboratory Method"` only.

**Pool-size check done before committing to a download count**: 4356 raw
records / 3256 distinct genomes with a real lab-measured vancomycin
phenotype (2045 Susceptible / 2017 Resistant raw, before dropping blank/
Intermediate/conflicting calls) -- a large, well-balanced pool used in
full, no subsampling needed.

Features: `sp_gene` collection, `property="Antibiotic Resistance"` -- same
raw BLAT-based CARD/NDARO/PATRIC gene-presence hits as every other
extension. Expected to be dominated by vanA/vanB cluster gene presence --
the clean, mechanistically-unambiguous glycopeptide-resistance signature
this species is named for testing (the mecA-style validation case, per
the task's own framing).

Grouping metadata: `genome` collection's `mlst` + `bioproject_accession`,
grouped split from the start.
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
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "bvbrc_efaecium"
DEFAULT_FEATURES_OUT = PROJECT_ROOT / "data" / "processed" / "efaecium_features.csv"
DEFAULT_METADATA_OUT = PROJECT_ROOT / "data" / "processed" / "efaecium_metadata.csv"

API_BASE = "https://www.bv-brc.org/api"
TAXON_ID = 1352  # Enterococcus faecium
ANTIBIOTIC = "vancomycin"
LABEL_COL = "is_resistant"
BATCH_SIZE = 150
SP_GENE_ROW_LIMIT = 25000
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


def fetch_vancomycin_labels() -> list:
    query = (
        f"and(eq(taxon_id,{TAXON_ID}),eq(antibiotic,{ANTIBIOTIC}),"
        f'eq(evidence,"Laboratory Method"))&limit(10000,0)'
    )
    return _rql_get("genome_amr", query)


def resolve_labels(records: list) -> dict:
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

    print(f"Resolved {len(labels)} genomes with a single, unambiguous vancomycin phenotype.")
    if dropped_conflicting:
        print(f"  Dropped {len(dropped_conflicting)} genome(s) with conflicting R/S calls.")
    return labels


def _batched(items, size):
    items = list(items)
    for i in range(0, len(items), size):
        yield items[i : i + size]


def fetch_genome_metadata(genome_ids: list) -> dict:
    metadata = {}
    batches = list(_batched(genome_ids, BATCH_SIZE))
    for i, batch in enumerate(batches, 1):
        print(f"  genome metadata batch {i}/{len(batches)}...", flush=True)
        ids = ",".join(batch)
        query = f"and(in(genome_id,({ids})))&select(genome_id,mlst,bioproject_accession,genome_name)&limit({BATCH_SIZE},0)"
        for row in _rql_get("genome", query):
            metadata[row["genome_id"]] = {
                "mlst": row.get("mlst", ""),
                "bioproject_accession": row.get("bioproject_accession", ""),
                "genome_name": row.get("genome_name", ""),
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
                f"sp_gene batch hit the row limit ({SP_GENE_ROW_LIMIT}) -- results likely "
                f"truncated. Raise SP_GENE_ROW_LIMIT and/or shrink BATCH_SIZE."
            )
        for row in rows:
            gid = row["genome_id"]
            features.setdefault(gid, {})[_feature_key(row)] = 1
        if checkpoint_path:
            checkpoint_path.write_text(json.dumps(features))
    n_with_features = sum(1 for v in features.values() if v)
    print(f"Fetched sp_gene AMR features for {n_with_features}/{len(genome_ids)} genomes.")
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
        meta_rows.append({
            "strain": gid, "ST": m.get("mlst", ""),
            "bioproject_accession": m.get("bioproject_accession", ""),
            "genome_name": m.get("genome_name", ""),
        })
    metadata_df = pd.DataFrame(meta_rows)
    return features_df, metadata_df, feature_names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-out", default=str(DEFAULT_FEATURES_OUT))
    parser.add_argument("--metadata-out", default=str(DEFAULT_METADATA_OUT))
    parser.add_argument("--cache-dir", default=str(RAW_DIR))
    parser.add_argument("--force-refetch", action="store_true")
    args = parser.parse_args()

    try:
        cache_dir = Path(args.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        labels_cache = cache_dir / "vancomycin_amr_lab.json"
        if labels_cache.exists() and not args.force_refetch:
            print(f"Reusing cached AMR records: {labels_cache}")
            raw_records = json.loads(labels_cache.read_text())
        else:
            print(f"Fetching E. faecium {ANTIBIOTIC} AMR records (Laboratory Method only) from BV-BRC...")
            raw_records = fetch_vancomycin_labels()
            print(f"  {len(raw_records)} raw records fetched, "
                  f"{len(set(r['genome_id'] for r in raw_records))} distinct genomes "
                  f"(before dropping blank/Intermediate/conflicting phenotypes)")
            labels_cache.write_text(json.dumps(raw_records))

        pheno_counts = Counter(r.get("resistant_phenotype", "(blank)") for r in raw_records)
        print(f"  Raw phenotype distribution: {dict(pheno_counts)}")

        labels = resolve_labels(raw_records)
        genome_ids = sorted(labels.keys())

        metadata_cache = cache_dir / "genome_metadata.json"
        if metadata_cache.exists() and not args.force_refetch:
            print(f"\nReusing cached genome metadata ({metadata_cache})...")
            metadata = json.loads(metadata_cache.read_text())
        else:
            print("\nFetching genome metadata (ST, BioProject) for grouped-split support...")
            metadata = fetch_genome_metadata(genome_ids)
            metadata_cache.write_text(json.dumps(metadata, indent=2))

        features_cache = cache_dir / "sp_gene_features.json"
        if features_cache.exists() and not args.force_refetch:
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

        print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
        print(f"Total genomes (unambiguous vancomycin phenotype): {n}")
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

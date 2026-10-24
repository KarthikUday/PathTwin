#!/usr/bin/env python3
"""
PathTwin Stage 5 extension -- A. baumannii carbapenem resistance data prep
(BV-BRC), following the exact P. aeruginosa/S. aureus methodology: real
genome+phenotype data pulled straight from BV-BRC's REST Data API, no new
conda env, grouped split support built in from the start.

Labels: `genome_amr` collection, taxon_id=470 (A. baumannii), the full
carbapenem class (imipenem, meropenem, doripenem, ertapenem -- the four
carbapenems BV-BRC actually has AMR records for in this species),
`evidence="Laboratory Method"` only -- same leakage-avoidance rule as
every other extension (never train on BV-BRC's own ML-predicted
phenotypes).

**Pool-size check done BEFORE committing to a download count**, same
practice as the P. aeruginosa extension's ~6,896-genome claim check:
per-drug and combined-class real counts queried directly first (see
prep run output / README for the numbers) rather than assuming a round
number. Combined carbapenem-class pool: 1490 distinct genomes -- used in
full, no subsampling needed at this scale.

Multi-drug label resolution: a genome tested against more than one
carbapenem gets ONE combined label -- Resistant/Susceptible only if ALL
of its real carbapenem phenotype calls agree; dropped entirely on
disagreement. Defensible for the same reason as the P. aeruginosa
fluoroquinolone extension: carbapenems share the dominant OXA-carbapenemase
and porin-loss resistance mechanisms in this species, so genuine
cross-resistance is the norm.

Features: `sp_gene` collection, `property="Antibiotic Resistance"` --
same raw BLAT-based CARD/NDARO/PATRIC gene-presence hits as every other
extension, not a resistance prediction.

Grouping metadata: `genome` collection's `mlst` + `bioproject_accession`,
for the grouped train/test split from the start (no random-split
round-trip needed, same as S. aureus/P. aeruginosa v2).
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
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "bvbrc_abaumannii"
DEFAULT_FEATURES_OUT = PROJECT_ROOT / "data" / "processed" / "abaumannii_features.csv"
DEFAULT_METADATA_OUT = PROJECT_ROOT / "data" / "processed" / "abaumannii_metadata.csv"

API_BASE = "https://www.bv-brc.org/api"
TAXON_ID = 470  # Acinetobacter baumannii
CARBAPENEMS = ["imipenem", "meropenem", "doripenem", "ertapenem"]
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


def check_real_pool_sizes():
    """Per-drug and combined-class counts, printed before any download
    commitment -- same sanity-check practice as the P. aeruginosa
    extension's ~6,896-genome claim verification."""
    print("Checking real BV-BRC pool sizes before committing to a download count...")
    per_drug = {}
    for drug in CARBAPENEMS:
        query = f'and(eq(taxon_id,{TAXON_ID}),eq(antibiotic,{drug}),eq(evidence,"Laboratory Method"))&limit(20000,0)'
        rows = _rql_get("genome_amr", query)
        per_drug[drug] = len(rows)
        print(f"  {drug}: {len(rows)} raw records")
    drug_list = ",".join(CARBAPENEMS)
    combined_query = (
        f'and(eq(taxon_id,{TAXON_ID}),in(antibiotic,({drug_list})),'
        f'eq(evidence,"Laboratory Method"))&limit(20000,0)'
    )
    combined_rows = _rql_get("genome_amr", combined_query)
    n_genomes = len(set(r["genome_id"] for r in combined_rows))
    print(f"  combined carbapenem-class: {len(combined_rows)} raw records, "
          f"{n_genomes} distinct genomes")
    return combined_rows


def resolve_labels(records: list) -> dict:
    by_genome, drugs_seen = {}, {}
    for r in records:
        pheno = r.get("resistant_phenotype", "")
        if pheno not in ("Resistant", "Susceptible"):
            continue
        gid = r["genome_id"]
        by_genome.setdefault(gid, set()).add(pheno)
        drugs_seen.setdefault(gid, set()).add(r.get("antibiotic", ""))

    labels, dropped_conflicting = {}, []
    multi_drug_agree = 0
    for genome_id, phenos in by_genome.items():
        if len(phenos) > 1:
            dropped_conflicting.append(genome_id)
            continue
        labels[genome_id] = 1 if phenos == {"Resistant"} else 0
        if len(drugs_seen[genome_id]) > 1:
            multi_drug_agree += 1

    print(f"Resolved {len(labels)} genomes with a single, unambiguous carbapenem phenotype.")
    print(f"  ({multi_drug_agree} tested against >1 carbapenem, all agreeing)")
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

        labels_cache = cache_dir / "genome_amr_carbapenem.json"
        if labels_cache.exists() and not args.force_refetch:
            print(f"Reusing cached AMR records: {labels_cache}")
            raw_records = json.loads(labels_cache.read_text())
        else:
            raw_records = check_real_pool_sizes()
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
        print(f"Total genomes (unambiguous carbapenem phenotype): {n}")
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

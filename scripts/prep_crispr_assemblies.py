#!/usr/bin/env python3
"""
PathTwin Stage 6 scale-up -- generic genome assembly downloader for
CRISPR-Cas scanning, used identically for A. baumannii, E. faecium, and
E. cloacae (same "one script, many species" precedent as crispr_scan.py,
train_bvbrc_classifier.py) rather than three near-duplicate files.

Verified directly before writing this that all three species are
API-only for Stage 5's data (BV-BRC sp_gene/genome_amr/genome JSON only,
no genome FASTA anywhere on disk) -- the exact same gap S. aureus had.
Flagged to the user before downloading anything; this script implements
whatever counts they chose per species.

Reuses the exact same BV-BRC genome_sequence-reconstruction method as
prep_saureus_crispr_assemblies.py / prep_paeruginosa_bvbrc_data.py (FTP
being unreachable from this environment). Reuses each species' own
already-cached, already-labeled genome pool from its Stage 5 prep script
-- no new label fetch, just assembly download for the chosen genome_ids.
"""
import argparse
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
API_BASE = "https://www.bv-brc.org/api"
SEQ_ROW_LIMIT = 500
REQUEST_TIMEOUT = 90
HEADERS = {"Accept": "application/json"}
RANDOM_SEED = 42
DOWNLOAD_WORKERS = 24


def _rql_get(path, query, retries=3):
    url = f"{API_BASE}/{path}/?{query}"
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, json.JSONDecodeError) as e:
            if attempt == retries - 1:
                raise RuntimeError(f"BV-BRC API request failed after {retries} attempts: {url}\n{e}")
            time.sleep(2 * (attempt + 1))
    return []


def download_assembly(genome_id, out_path, size_min, size_max):
    query = f"eq(genome_id,{genome_id})&select(sequence,length,accession)&limit({SEQ_ROW_LIMIT},0)"
    try:
        rows = _rql_get("genome_sequence", query)
    except RuntimeError as e:
        print(f"    fetch failed for {genome_id}: {e}")
        return False
    if not rows:
        return False
    if len(rows) >= SEQ_ROW_LIMIT:
        print(f"    WARNING: {genome_id} hit row cap {SEQ_ROW_LIMIT}, skipping (likely truncated)")
        return False
    total_len = sum(r.get("length", 0) or 0 for r in rows)
    if not (size_min <= total_len <= size_max):
        return False
    with open(out_path, "w") as fh:
        for i, r in enumerate(rows, 1):
            acc = r.get("accession") or f"contig_{i}"
            seq = r.get("sequence", "")
            if not seq:
                continue
            fh.write(f">{acc}\n")
            for j in range(0, len(seq), 70):
                fh.write(seq[j : j + 70] + "\n")
    return True


def _download_one(genome_id, assembly_dir, size_min, size_max):
    out_path = assembly_dir / f"{genome_id}.fna"
    if out_path.exists() and out_path.stat().st_size > 0:
        return genome_id, True
    ok = download_assembly(genome_id, out_path, size_min, size_max)
    return genome_id, ok


def resolve_labels_from_amr_json(records, drug_filter=None):
    """Same resolve-one-label-per-genome rule as every prep script:
    unambiguous Resistant/Susceptible only."""
    by_genome = {}
    for r in records:
        pheno = r.get("resistant_phenotype", "")
        if pheno not in ("Resistant", "Susceptible"):
            continue
        gid = r["genome_id"]
        by_genome.setdefault(gid, set()).add(pheno)
    return {gid: (1 if phenos == {"Resistant"} else 0)
            for gid, phenos in by_genome.items() if len(phenos) == 1}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--species-tag", required=True, help="e.g. abaumannii, efaecium, ecloacae")
    ap.add_argument("--labels-cache", required=True, help="path to the species' cached genome_amr_*.json")
    ap.add_argument("--verified-ids-json", default=None,
                     help="optional: path to a JSON with a 'verified' list restricting eligible genome_ids "
                          "(E. cloacae's species_verification.json)")
    ap.add_argument("--assembly-dir", required=True)
    ap.add_argument("--size-min", type=int, required=True)
    ap.add_argument("--size-max", type=int, required=True)
    ap.add_argument("--n-target", type=int, default=None, help="omit for the full resolved pool")
    args = ap.parse_args()

    assembly_dir = Path(args.assembly_dir)
    assembly_dir.mkdir(parents=True, exist_ok=True)

    records = json.loads(Path(args.labels_cache).read_text())
    labels = resolve_labels_from_amr_json(records)

    if args.verified_ids_json:
        verified = set(json.loads(Path(args.verified_ids_json).read_text())["verified"])
        before = len(labels)
        labels = {gid: y for gid, y in labels.items() if gid in verified}
        print(f"Restricted to species-verified genomes: {before} -> {len(labels)}")

    resistant = [g for g, y in labels.items() if y == 1]
    susceptible = [g for g, y in labels.items() if y == 0]
    print(f"[{args.species_tag}] Labeled pool: {len(labels)} genomes ({len(resistant)} R, {len(susceptible)} S)")

    rng = random.Random(RANDOM_SEED)
    rng.shuffle(resistant)
    rng.shuffle(susceptible)

    if args.n_target is None or args.n_target >= len(labels):
        sample = resistant + susceptible
        print(f"[{args.species_tag}] Using FULL pool: {len(sample)} genomes")
    else:
        frac_r = len(resistant) / len(labels)
        n_r = round(args.n_target * frac_r)
        n_s = args.n_target - n_r
        sample = resistant[:n_r] + susceptible[:n_s]
        print(f"[{args.species_tag}] Sampling {len(sample)} genomes ({n_r} R, {n_s} S) "
              f"stratified to match pool ratio ({frac_r:.1%} R)")
    rng.shuffle(sample)

    sample_labels_path = assembly_dir.parent / "crispr_sample_labels.json"
    sample_labels_path.write_text(json.dumps({g: labels[g] for g in sample}, indent=2))

    ok_count, fail_count = 0, 0
    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as ex:
        futures = {ex.submit(_download_one, gid, assembly_dir, args.size_min, args.size_max): gid for gid in sample}
        done = 0
        for fut in as_completed(futures):
            gid, ok = fut.result()
            done += 1
            if ok:
                ok_count += 1
            else:
                fail_count += 1
            if done % 25 == 0 or done == len(sample):
                print(f"  {done}/{len(sample)} attempted ({ok_count} ok, {fail_count} failed)", flush=True)

    print(f"\n[{args.species_tag}] Done. {ok_count} assemblies downloaded to {assembly_dir}, {fail_count} failed/skipped.")


if __name__ == "__main__":
    main()

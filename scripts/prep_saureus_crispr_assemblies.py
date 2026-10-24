#!/usr/bin/env python3
"""
PathTwin Stage 6 scale-up -- S. aureus genome assembly download for CRISPR-Cas
scanning.

Stage 5's S. aureus oxacillin classifier (prep_saureus_bvbrc_data.py) never
downloaded genome assemblies -- it pulled gene-presence features directly
from BV-BRC's sp_gene REST collection. There is therefore no existing
assembly set to "reuse" for this species; this script fills that gap using
the same genome_sequence-reconstruction method prep_paeruginosa_bvbrc_data.py
used (BV-BRC FTP being unreachable from this environment previously).

Reuses the already-cached, already-labeled genome pool from Stage 5
(data/raw/bvbrc_saureus/oxacillin_amr_lab.json + genome_metadata.json) --
no new label fetch, just assembly download for a random subsample of that
pool, stratified to keep the R/S balance close to the full pool's 41.1%
resistant.
"""
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

PROJECT_ROOT = Path("/home/kell/PathTwin")
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "bvbrc_saureus"
ASSEMBLY_DIR = RAW_DIR / "assemblies"
API_BASE = "https://www.bv-brc.org/api"
SEQ_ROW_LIMIT = 500
REQUEST_TIMEOUT = 90
HEADERS = {"Accept": "application/json"}
RANDOM_SEED = 42
DOWNLOAD_WORKERS = 24
N_TARGET = 400  # target sample size


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


def download_assembly(genome_id, out_path):
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
    # S. aureus genomes are ~2.7-3.0 Mbp; sanity bound with margin
    if not (2_400_000 <= total_len <= 3_400_000):
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


def _download_one(genome_id):
    out_path = ASSEMBLY_DIR / f"{genome_id}.fna"
    if out_path.exists() and out_path.stat().st_size > 0:
        return genome_id, True
    ok = download_assembly(genome_id, out_path)
    return genome_id, ok


def main():
    ASSEMBLY_DIR.mkdir(parents=True, exist_ok=True)

    labels_cache = RAW_DIR / "oxacillin_amr_lab.json"
    records = json.loads(labels_cache.read_text())

    # resolve one label per genome (same rule as prep_saureus_bvbrc_data.py:
    # keep if unambiguous Resistant/Susceptible)
    by_genome = {}
    for r in records:
        pheno = r.get("resistant_phenotype", "")
        if pheno not in ("Resistant", "Susceptible"):
            continue
        gid = r["genome_id"]
        by_genome.setdefault(gid, set()).add(pheno)
    labels = {gid: (1 if phenos == {"Resistant"} else 0)
              for gid, phenos in by_genome.items() if len(phenos) == 1}

    resistant = [g for g, y in labels.items() if y == 1]
    susceptible = [g for g, y in labels.items() if y == 0]
    print(f"Labeled pool: {len(labels)} genomes ({len(resistant)} R, {len(susceptible)} S)")

    rng = random.Random(RANDOM_SEED)
    rng.shuffle(resistant)
    rng.shuffle(susceptible)
    frac_r = len(resistant) / len(labels)
    n_r = round(N_TARGET * frac_r)
    n_s = N_TARGET - n_r
    sample = resistant[:n_r] + susceptible[:n_s]
    rng.shuffle(sample)
    print(f"Sampling {len(sample)} genomes ({n_r} R, {n_s} S) stratified to match pool ratio ({frac_r:.1%} R)")

    sample_labels_path = RAW_DIR / "crispr_sample_labels.json"
    sample_labels_path.write_text(json.dumps({g: labels[g] for g in sample}, indent=2))

    ok_count, fail_count = 0, 0
    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as ex:
        futures = {ex.submit(_download_one, gid): gid for gid in sample}
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

    print(f"\nDone. {ok_count} assemblies downloaded to {ASSEMBLY_DIR}, {fail_count} failed/skipped.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
PathTwin Stage 5 extension -- P. aeruginosa fluoroquinolone-resistance data
prep (BV-BRC), with real genome assemblies (not just annotation tables).

v2 -- scaled up from the first run's 179-genome subsample. That run tested
whether adding 5-locus regulatory SNP features beats gene-presence-only
prediction and found real SHAP-visible signal (mexZ_missense, oprD_lof)
but no clear aggregate accuracy/AUC win; this pull maximizes the real,
Laboratory-Method-only sample size to test whether that holds at scale.

**Pool-size discrepancy, verified and reported rather than silently
matched**: the task asked to "aim for as much of the ~6,896-genome pool"
as possible. Directly queried BV-BRC before writing any code against that
number and could not reproduce it under any real-phenotype-only
(`evidence="Laboratory Method"`) query for this species: ciprofloxacin
alone is 1175 raw records / 662 distinct genomes (unchanged from the first
run); the full fluoroquinolone class (ciprofloxacin + levofloxacin +
moxifloxacin + norfloxacin + ofloxacin) is 1821 raw records / 1147
distinct genomes; ALL antibiotics combined, Laboratory Method only, is
11,121 records; ALL P. aeruginosa genomes in BV-BRC regardless of
phenotype data is 16,026. None of these is ~6,896. Rather than relax the
leakage-avoidance rule (e.g. including BV-BRC's own computational-method
predictions, which would trivially inflate the count but reintroduce
exactly the leakage risk this project has excluded from every prior
extension) to hit an unverifiable target, this pull uses the full
**fluoroquinolone class** pool (1147 genomes) -- broader than the first
run's ciprofloxacin-only pool (662), still real lab-measured phenotypes
only, and a genuine, defensible way to maximize real sample size. This is
within the original task's own framing ("ciprofloxacin/fluoroquinolone
AMR phenotype data") -- the first run narrowed to ciprofloxacin alone for
a single-clean-antibiotic baseline; this run broadens back out
specifically because more real data is the explicit goal here.

Multi-drug label resolution: a genome tested against more than one
fluoroquinolone gets ONE combined label -- Resistant/Susceptible only if
ALL of its real fluoroquinolone phenotype calls agree; dropped entirely
(same as a same-drug R/S conflict) if they disagree. This is scientifically
defensible, not just convenient: fluoroquinolones share the dominant
target-site (GyrA/ParC) and efflux resistance mechanisms, so genuine
cross-resistance is the norm and a disagreement more likely reflects a
borderline MIC near one drug's specific breakpoint or a data-quality issue
than two truly independent phenotypes.

BV-BRC API gotcha from the first run, NOT repeated here: RQL queries
default to a **25-row limit** with no warning if `limit(N)` is omitted --
confirmed then by comparing `content-range` headers on a known-71-contig
genome (25 rows silently returned with no limit). Every query in this
script, including the `genome_sequence` per-genome fetch, passes an
explicit limit for exactly this reason.

Assembly download is parallelized this time (ThreadPoolExecutor, I/O-bound
work) -- the first run's fully sequential download was the single slowest
part of that pipeline (~180 genomes took the better part of an hour).
"""

import argparse
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "bvbrc_paeruginosa"
ASSEMBLY_DIR = RAW_DIR / "assemblies"
DEFAULT_METADATA_OUT = PROJECT_ROOT / "data" / "processed" / "paeruginosa_metadata.csv"

API_BASE = "https://www.bv-brc.org/api"
TAXON_ID = 287  # Pseudomonas aeruginosa
FLUOROQUINOLONES = ["ciprofloxacin", "levofloxacin", "moxifloxacin", "norfloxacin", "ofloxacin"]
BATCH_SIZE = 150
REQUEST_TIMEOUT = 90
SEQ_ROW_LIMIT = 2000  # generously above any real assembly's contig count
HEADERS = {"Accept": "application/json"}
RANDOM_SEED = 42
DOWNLOAD_WORKERS = 24


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


def fetch_fluoroquinolone_labels() -> list:
    ab_list = ",".join(FLUOROQUINOLONES)
    query = (
        f"and(eq(taxon_id,{TAXON_ID}),in(antibiotic,({ab_list})),"
        f'eq(evidence,"Laboratory Method"))&limit(5000,0)'
    )
    return _rql_get("genome_amr", query)


def resolve_labels(records: list) -> dict:
    """Collapse per-genome records into a single label across all real
    fluoroquinolone phenotype calls for that genome (possibly several
    different drugs); drop blank/Intermediate phenotypes and genomes with
    conflicting R/S calls (see module docstring for why this is a
    defensible policy, not just convenience)."""
    by_genome = {}
    drugs_seen = {}
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

    print(f"Resolved {len(labels)} genomes with a single, unambiguous fluoroquinolone phenotype.")
    print(f"  ({multi_drug_agree} of those were tested against >1 fluoroquinolone, all agreeing)")
    if dropped_conflicting:
        print(f"  Dropped {len(dropped_conflicting)} genome(s) with conflicting R/S calls "
              f"(same drug across records, or disagreement across different fluoroquinolones).")
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
        query = f"and(in(genome_id,({ids})))&select(genome_id,mlst,bioproject_accession,genome_name)&limit({BATCH_SIZE},0)"
        for row in _rql_get("genome", query):
            metadata[row["genome_id"]] = {
                "mlst": row.get("mlst", ""),
                "bioproject_accession": row.get("bioproject_accession", ""),
                "genome_name": row.get("genome_name", ""),
            }
    print(f"Fetched metadata for {len(metadata)}/{len(genome_ids)} genomes.")
    return metadata


def download_assembly(genome_id: str, out_path: Path) -> bool:
    """Reconstruct a FASTA assembly from the genome_sequence collection's
    per-contig `sequence` field. Returns False (does not write) if the
    genome has no sequence data or is implausibly small/large for a real
    P. aeruginosa genome (sanity bound: 4-9 Mbp)."""
    query = f"eq(genome_id,{genome_id})&select(sequence,length,accession)&limit({SEQ_ROW_LIMIT},0)"
    try:
        rows = _rql_get("genome_sequence", query)
    except RuntimeError as e:
        print(f"    fetch failed for {genome_id}: {e}")
        return False

    if not rows:
        return False
    if len(rows) >= SEQ_ROW_LIMIT:
        raise RuntimeError(
            f"genome_sequence row count for {genome_id} hit the {SEQ_ROW_LIMIT} "
            f"safety cap -- likely truncated, raise SEQ_ROW_LIMIT and re-run."
        )

    total_len = sum(r.get("length", 0) for r in rows)
    if not (4_000_000 <= total_len <= 9_000_000):
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


def _download_one(genome_id: str):
    out_path = ASSEMBLY_DIR / f"{genome_id}.fna"
    if out_path.exists() and out_path.stat().st_size > 0:
        return genome_id, True
    ok = download_assembly(genome_id, out_path)
    return genome_id, ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-genomes", type=int, default=None,
                         help="Cap the number of genomes attempted (default: no cap -- attempt the full resolved pool)")
    parser.add_argument("--metadata-out", type=Path, default=DEFAULT_METADATA_OUT)
    parser.add_argument("--force-refetch", action="store_true",
                         help="Re-fetch labels/metadata even if cached raw JSON exists")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    ASSEMBLY_DIR.mkdir(parents=True, exist_ok=True)

    labels_cache = RAW_DIR / "genome_amr_fluoroquinolone.json"
    if labels_cache.exists() and not args.force_refetch:
        print(f"Reusing cached AMR records: {labels_cache}")
        records = json.loads(labels_cache.read_text())
    else:
        print(f"Fetching fluoroquinolone-class AMR records from BV-BRC "
              f"({', '.join(FLUOROQUINOLONES)})...")
        records = fetch_fluoroquinolone_labels()
        labels_cache.write_text(json.dumps(records))
        print(f"  {len(records)} raw records fetched, cached to {labels_cache}")

    labels = resolve_labels(records)
    all_genome_ids = sorted(labels)

    meta_cache = RAW_DIR / "genome_metadata_v2.json"
    if meta_cache.exists() and not args.force_refetch:
        print(f"Reusing cached genome metadata: {meta_cache}")
        metadata = json.loads(meta_cache.read_text())
    else:
        print("Fetching genome metadata (ST, BioProject)...")
        metadata = fetch_genome_metadata(all_genome_ids)
        meta_cache.write_text(json.dumps(metadata))

    # Unbiased order -- fixed seed shuffle, no filtering by resistance-gene
    # content or phenotype (see module docstring). With no --max-genomes
    # cap, order doesn't matter for the final set, only for progress
    # reporting; kept for parity with a capped run.
    rng = random.Random(RANDOM_SEED)
    pool = list(all_genome_ids)
    rng.shuffle(pool)
    if args.max_genomes:
        pool = pool[: args.max_genomes]

    resistant_n = sum(1 for g in all_genome_ids if labels[g] == 1)
    print(f"\nFull resolved pool: {len(all_genome_ids)} genomes "
          f"({resistant_n} resistant / {len(all_genome_ids) - resistant_n} susceptible)")
    print(f"Attempting assembly download for {len(pool)} genomes, "
          f"{DOWNLOAD_WORKERS} parallel workers...")

    selected, skipped_no_assembly = [], []
    done = 0
    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as executor:
        futures = {executor.submit(_download_one, gid): gid for gid in pool}
        for future in as_completed(futures):
            gid = futures[future]
            done += 1
            try:
                _, ok = future.result()
            except Exception as e:
                print(f"    exception for {gid}: {e}")
                ok = False
            if ok:
                selected.append(gid)
            else:
                skipped_no_assembly.append(gid)
            if done % 100 == 0 or done == len(pool):
                print(f"  {done}/{len(pool)} attempted, {len(selected)} usable so far...", flush=True)

    print(f"\nSelected {len(selected)} genomes with usable assemblies "
          f"({len(skipped_no_assembly)} skipped -- no/implausible sequence data).")

    sel_resistant = sum(1 for g in selected if labels[g] == 1)
    print(f"Selected sample class balance: {sel_resistant} resistant / "
          f"{len(selected) - sel_resistant} susceptible "
          f"({sel_resistant / len(selected) * 100:.1f}% resistant)")

    rows = []
    for genome_id in selected:
        m = metadata.get(genome_id, {})
        rows.append({
            "genome_id": genome_id,
            "genome_name": m.get("genome_name", ""),
            "is_resistant": labels[genome_id],
            "mlst": m.get("mlst", ""),
            "bioproject_accession": m.get("bioproject_accession", ""),
            "assembly_path": str((ASSEMBLY_DIR / f"{genome_id}.fna").relative_to(PROJECT_ROOT)),
        })
    df = pd.DataFrame(rows)
    args.metadata_out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.metadata_out, index=False)
    print(f"\nWrote metadata for {len(df)} genomes to {args.metadata_out}")

    n_st_groups = df["mlst"].replace("", pd.NA).nunique()
    n_bp_groups = df["bioproject_accession"].replace("", pd.NA).nunique()
    print(f"Grouping columns available: {n_st_groups} distinct ST values, "
          f"{n_bp_groups} distinct BioProject values")


if __name__ == "__main__":
    main()

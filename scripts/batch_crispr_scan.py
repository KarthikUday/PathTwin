#!/usr/bin/env python3
"""
PathTwin Stage 6 scale-up -- parallel batch driver for crispr_scan.py.

Runs cctyper across many genome assemblies in parallel (ProcessPoolExecutor
of individual `cctyper` subprocess calls. each cctyper invocation is
itself lightly multi-threaded via --threads). Resumable: skips genomes that
already have a cached summary JSON, matching crispr_scan.py's own
already_ran cache check.

Must run inside the pathtwin-crispr conda env (needs cctyper on PATH).
"""
import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, "/home/kell/PathTwin/scripts")
from crispr_scan import genome_to_crispr_profile, RESULTS_DIR


def _run_one(fasta_path_str, threads):
    fasta_path = Path(fasta_path_str)
    genome_id = fasta_path.stem
    summary_path = RESULTS_DIR / f"{genome_id}_summary.json"
    if summary_path.exists():
        return genome_id, "cached", None
    try:
        profile = genome_to_crispr_profile(fasta_path, threads=threads)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(profile, f, indent=2)
        return genome_id, "ok", None
    except Exception as e:
        return genome_id, "error", str(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta-dir", required=True)
    ap.add_argument("--pattern", default="*.fna")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--threads-per-job", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    fasta_dir = Path(args.fasta_dir)
    fastas = sorted(fasta_dir.glob(args.pattern))
    if args.limit:
        fastas = fastas[: args.limit]
    print(f"{len(fastas)} genome(s) to process from {fasta_dir}", flush=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    n_ok, n_err, n_cached = 0, 0, 0
    errors = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_run_one, str(f), args.threads_per_job): f for f in fastas}
        done = 0
        for fut in as_completed(futures):
            genome_id, status, err = fut.result()
            done += 1
            if status == "ok":
                n_ok += 1
            elif status == "cached":
                n_cached += 1
            else:
                n_err += 1
                errors.append((genome_id, err))
                print(f"  ERROR {genome_id}: {err}", flush=True)
            if done % 10 == 0 or done == len(fastas):
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                remaining = (len(fastas) - done) / rate if rate > 0 else float("nan")
                print(f"  [{done}/{len(fastas)}] ok={n_ok} cached={n_cached} err={n_err} "
                      f"elapsed={elapsed/60:.1f}min ETA={remaining/60:.1f}min", flush=True)

    print(f"\nDONE. ok={n_ok} cached={n_cached} err={n_err} total={len(fastas)} "
          f"in {(time.time()-t0)/60:.1f} min", flush=True)
    if errors:
        print(f"\n{len(errors)} error(s):", flush=True)
        for gid, err in errors[:30]:
            print(f"  {gid}: {err[:200]}", flush=True)


if __name__ == "__main__":
    main()

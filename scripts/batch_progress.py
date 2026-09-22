#!/usr/bin/env python3
"""
PathTwin Stage 6 v2/v3 -- read-only progress display for the running
CRISPR-Cas batch scans (P. aeruginosa/S. aureus, then A. baumannii/
E. faecium/E. cloacae).

Only inspects existing files under results/crispr_scan/ and the input
assembly directories -- never touches the running batch jobs, their
processes, or cctyper's own output. Safe to run at any time, as many
times as you like, alongside the live scans.

Completion rate (for the ETA estimate) is derived from the mtimes of the
summary JSONs already written -- earliest mtime approximates batch start,
most recent mtime approximates "now" for the purposes of the rate, so the
estimate is a genome per minute rate that stays accurate over the run.
"""
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path("/home/kell/PathTwin")
CRISPR_DIR = PROJECT_ROOT / "results" / "crispr_scan"

SPECIES = [
    ("P. aeruginosa", PROJECT_ROOT / "data/raw/bvbrc_paeruginosa/assemblies", "*.fna"),
    ("S. aureus", PROJECT_ROOT / "data/raw/bvbrc_saureus/assemblies", "*.fna"),
    ("A. baumannii", PROJECT_ROOT / "data/raw/bvbrc_abaumannii/assemblies", "*.fna"),
    ("E. faecium", PROJECT_ROOT / "data/raw/bvbrc_efaecium/assemblies", "*.fna"),
    ("E. cloacae", PROJECT_ROOT / "data/raw/bvbrc_ecloacae/assemblies", "*.fna"),
]

BAR_WIDTH = 40


def fmt_eta(minutes):
    if minutes != minutes or minutes < 0:  # NaN guard
        return "unknown"
    if minutes < 60:
        return f"{minutes:.0f} min"
    h = int(minutes // 60)
    m = int(minutes % 60)
    return f"{h}h {m}m"


def progress_bar(frac, width=BAR_WIDTH):
    filled = int(round(frac * width))
    return "[" + "█" * filled + "░" * (width - filled) + "]"


def report_species(label, fasta_dir, pattern):
    fastas = sorted(fasta_dir.glob(pattern))
    total = len(fastas)
    if total == 0:
        print(f"{label}: no assemblies found in {fasta_dir}")
        return

    mtimes = []
    done = 0
    for f in fastas:
        summary = CRISPR_DIR / f"{f.stem}_summary.json"
        if summary.exists():
            done += 1
            mtimes.append(summary.stat().st_mtime)

    frac = done / total
    print(f"\n{label}")
    print(f"  {progress_bar(frac)} {frac*100:5.1f}%")
    print(f"  {done}/{total} genomes completed")

    if done >= 2:
        earliest, latest = min(mtimes), max(mtimes)
        elapsed_min = (latest - earliest) / 60.0
        if elapsed_min > 0:
            rate = done / elapsed_min  # genomes/min
            remaining = total - done
            eta_min = remaining / rate if rate > 0 else float("nan")
            since_last_min = (time.time() - latest) / 60.0
            print(f"  rate: {rate:.2f} genomes/min  |  ETA: {fmt_eta(eta_min)}"
                  f"  |  last completion: {since_last_min:.1f} min ago")
        else:
            print("  rate: (all completions within the same minute -- ETA not yet stable)")
    elif done == 1:
        since_last_min = (time.time() - mtimes[0]) / 60.0
        print(f"  rate: not enough completions yet for an estimate (last completion {since_last_min:.1f} min ago)")
    else:
        print("  rate: no genomes completed yet")


def main():
    print(f"PathTwin Stage 6 v2 -- batch scan progress ({time.strftime('%Y-%m-%d %H:%M:%S')})")
    for label, fasta_dir, pattern in SPECIES:
        report_species(label, fasta_dir, pattern)
    print()


if __name__ == "__main__":
    main()

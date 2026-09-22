#!/usr/bin/env python3
"""
PathTwin Stage 6 -- CRISPR-Cas phage-defense profiling

Runs CRISPRCasTyper (`cctyper`) on a genome assembly and extracts:
  - whether a CRISPR-Cas system is confidently present (a Cas operon that
    passed cctyper's own confidence thresholds -- not just a weak/candidate
    hit)
  - its subtype(s), specifically flagging type I-E: the literature ties
    I-E CRISPR-Cas to an inverse correlation with ESBL/acquired-resistance
    burden in Enterobacteriaceae
  - CRISPR array count -- both the raw count from minced and how many of
    those cctyper actually trusts (a repeat array can exist with a
    below-threshold subtype call, or with no confidently-linked Cas operon
    at all: an "orphan" array)

cctyper's own output schema, used directly rather than reinvented:
  - cas_operons.tab           confident Cas operon calls (only exists, and
                               only has rows, when at least one operon
                               passed cctyper's confidence thresholds)
  - cas_operons_putative.tab  candidate operons that did NOT pass -- always
                               written when nothing confident was found,
                               never treated as "presence" here
  - crisprs_all.tab           every CRISPR array minced found, each with
                               its own repeat-based subtype guess + whether
                               cctyper trusts that array's stats
  - crisprs_near_cas.tab      the subset of arrays within --ccd of a
                               confident Cas operon (i.e. actually part of
                               a real CRISPR-Cas system, not orphaned)
  - CRISPR_Cas.tab            the final combined call when a confident Cas
                               operon and an array are linked

Requires the `pathtwin-crispr` conda env (`cctyper` on PATH, its database
auto-downloaded by the conda package's post-link step into
`$CONDA_PREFIX/cct_data` -- set `CCTYPER_DB` if running from elsewhere).
"""

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "crispr_scan"

FLAGGED_SUBTYPE = "I-E"  # literature ties this specifically to ESBL correlation


def _read_tab(path: Path) -> list:
    """Read a cctyper tab-separated output file; [] if it doesn't exist
    (cctyper only writes some files when there's something to put in them)."""
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def genome_to_crispr_profile(genome_fasta_path, threads: int = 4, force_rerun: bool = False) -> dict:
    """
    Run cctyper on a single genome assembly and return its CRISPR-Cas
    profile summary.

    `force_rerun=True` re-runs cctyper even if cached output already
    exists for this genome (default: reuse the cache -- cctyper's
    prodigal/HMMER/minced/BLAST pipeline is the expensive step, and its
    output is deterministic for a given genome + database version).

    **Cache-completeness bug fixed 2026-09-04**: this used to treat
    `arguments.tab` (the FIRST file cctyper writes, before any real work
    happens) as the "already ran" signal. An interrupted run (e.g. a
    power loss mid-scan, which genuinely happened during this project's
    1420-genome batch run) leaves `arguments.tab` present but the rest of
    the output truncated/partial -- the old check would silently accept
    that as "done" and hand back a false-negative profile (empty
    cas_operons.tab misread as "no CRISPR-Cas") without ever re-running
    cctyper. Fixed by keying the cache on the summary JSON itself, which
    is only ever written after a full, successful parse -- the one file
    that can't exist for a truncated run.
    """
    genome_fasta_path = Path(genome_fasta_path)
    if not genome_fasta_path.exists():
        raise FileNotFoundError(f"Genome FASTA not found: {genome_fasta_path}")

    genome_id = genome_fasta_path.stem
    outdir = RESULTS_DIR / genome_id
    summary_path = RESULTS_DIR / f"{genome_id}_summary.json"
    already_ran = summary_path.exists()

    if already_ran and not force_rerun:
        with open(summary_path) as f:
            return json.load(f)

    # Reaching here means either force_rerun=True, or no valid summary
    # JSON exists yet for this genome (already_ran=False) -- either way,
    # a fresh cctyper run is needed. Any leftover directory (a truncated
    # prior run, or a stale complete one under force_rerun) is wiped
    # first, since cctyper refuses to write into an existing dir.
    if outdir.exists():
        shutil.rmtree(outdir)
    # cctyper creates its own output dir via a bare os.mkdir() (not
    # recursive) -- the parent must already exist, or it FileNotFoundErrors.
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        "cctyper", str(genome_fasta_path), str(outdir),
        "--prodigal", "single", "-t", str(threads),
        "--no_plot", "--simplelog",
    ]
    print(f"  Running cctyper on {genome_fasta_path.name}...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"cctyper failed on {genome_fasta_path}:\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    # Confident Cas operon calls -- cas_operons.tab only exists (and only
    # has rows) when at least one operon passed cctyper's confidence
    # thresholds. Its absence (only cas_operons_putative.tab present) means
    # no confident system, even if weak/candidate hits exist -- those are
    # not counted as "present" here.
    confident_operons = _read_tab(outdir / "cas_operons.tab")
    cas_system_present = len(confident_operons) > 0
    cas_subtypes = sorted({row["Best_type"] for row in confident_operons})

    all_arrays = _read_tab(outdir / "crisprs_all.tab")
    array_subtypes = sorted({row["Subtype"] for row in all_arrays if row.get("Subtype")})
    near_cas = _read_tab(outdir / "crisprs_near_cas.tab")

    return {
        "genome_id": genome_id,
        "cas_system_present": cas_system_present,
        "cas_subtypes": cas_subtypes,
        "is_type_I_E": FLAGGED_SUBTYPE in cas_subtypes,
        "n_arrays_total": len(all_arrays),
        "n_arrays_trusted": sum(1 for row in all_arrays if row.get("Trusted") == "True"),
        "array_subtypes": array_subtypes,
        "n_arrays_linked_to_cas": len(near_cas),
        "cctyper_outdir": str(outdir),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run CRISPRCasTyper on a genome and summarize its CRISPR-Cas profile."
    )
    parser.add_argument("--genome", required=True, help="Path to genome assembly FASTA")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--output-json", default=None,
        help="Path to write the summary JSON (default: results/crispr_scan/<genome_id>_summary.json)",
    )
    parser.add_argument("--force-rerun", action="store_true", help="Re-run cctyper even if cached output exists")
    args = parser.parse_args()

    try:
        profile = genome_to_crispr_profile(args.genome, threads=args.threads, force_rerun=args.force_rerun)
    except Exception as e:
        sys.exit(f"ERROR: {e}")

    out_path = Path(args.output_json) if args.output_json else RESULTS_DIR / f"{profile['genome_id']}_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(profile, f, indent=2)

    print(f"\nGenome: {profile['genome_id']}")
    print(f"  CRISPR-Cas system present : {profile['cas_system_present']}")
    print(f"  Subtype(s)                : {profile['cas_subtypes'] or '(none)'}")
    if profile["is_type_I_E"]:
        print(f"  *** Type I-E detected -- literature ties this to ESBL correlation ***")
    print(f"  CRISPR arrays (total)     : {profile['n_arrays_total']}  (trusted: {profile['n_arrays_trusted']})")
    print(f"  Array subtype call(s)     : {profile['array_subtypes'] or '(none)'}")
    print(f"  Arrays linked to a confident Cas operon : {profile['n_arrays_linked_to_cas']}")
    print(f"  Summary written to: {out_path}")


if __name__ == "__main__":
    main()

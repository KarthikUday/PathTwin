#!/usr/bin/env python3
"""
PathTwin Stage 5 extension -- P. aeruginosa fluoroquinolone resistance:
per-genome pipeline runner (RGI gene presence/absence + snippy regulatory
SNP calling), building both the baseline and combined feature matrices.

For each genome in data/processed/paeruginosa_metadata.csv (from
prep_paeruginosa_bvbrc_data.py), this script:

  1. Runs `rgi main` on the assembly to get a gene-presence/absence vector
     (same approach as build_feature_matrix.py / Stage 1, but using
     DIAMOND instead of BLAST -- see the "why DIAMOND" note below).
  2. Runs snippy (via scripts/mutation_scan.py's same conda-run-into-
     pathtwin-snippy pattern) against the PAO1 reference, then parses
     snps.tab for coding variants in exactly five regulatory/porin loci:
     mexR, nalC, nalD, mexZ (annotated as "nfxB" in PAO1's RefSeq record
     -- its older/alternate gene name; verified directly against
     data/raw/references/GCF_000006765.1_ASM676v1_genomic.gbff, not
     assumed), and oprD.
  3. Writes one row per genome to two output CSVs:
       data/processed/paeruginosa_features_baseline.csv  (RGI presence only)
       data/processed/paeruginosa_features_combined.csv  (RGI presence + SNP features)

Why DIAMOND, not BLAST, for RGI here: this project's every other Stage 1
RGI run (single genomes, one-off) used RGI's default BLAST aligner. Running
BLAST-based RGI on ~200 genomes was benchmarked at several minutes/genome --
tens of hours of total runtime, impractical for one session. DIAMOND
(`rgi main -a DIAMOND`) is RGI's own supported, documented alternative
aligner and was benchmarked on an identical genome (PA14) at ~16 seconds
vs. several minutes for BLAST -- but it IS an approximate/heuristic
aligner, and a direct hit-list comparison against this project's own
earlier BLAST-based PA14 result showed a handful of differences (54 vs 58
hits; DIAMOND missed a few weaker/marginal-homology hits BLAST caught, and
called one allele-level distinction differently: "Type A NfxB" vs "Type B
NfxB"). This is a real, documented sensitivity tradeoff made explicitly for
bulk-run tractability -- see README "P. aeruginosa fluoroquinolone
extension" for the full account. It does not apply to any of this
project's other Stage 1/Stage 2 single-genome RGI results, which all still
use BLAST.

SNP feature categories, per gene: two binary features,
"<gene>_missense" (any non-synonymous coding substitution or in-frame
indel that isn't loss-of-function) and "<gene>_lof" (frameshift_variant,
stop_gained, or start_lost -- true loss-of-function). oprD's LOF flag is
the specific, literature-established carbapenem/fluoroquinolone-
permeability mechanism the task calls out; the same LOF/missense split is
applied to the other four genes too since LOF of a repressor (mexR, nalC,
nalD, mexZ/nfxB) is *also* a real known de-repression mechanism for their
respective efflux pumps (MexAB-OprM for mexR/nalC/nalD; MexXY for mexZ),
not just a courtesy extension of the pattern.
"""

import argparse
import csv
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
METADATA_CSV = PROJECT_ROOT / "data" / "processed" / "paeruginosa_metadata.csv"
ASSEMBLY_DIR = PROJECT_ROOT / "data" / "raw" / "bvbrc_paeruginosa" / "assemblies"
RGI_WORKDIR = PROJECT_ROOT / "results" / "rgi_paeruginosa_bulk"
SNIPPY_WORKDIR = PROJECT_ROOT / "results" / "snippy_paeruginosa_bulk"
BASELINE_OUT = PROJECT_ROOT / "data" / "processed" / "paeruginosa_features_baseline.csv"
COMBINED_OUT = PROJECT_ROOT / "data" / "processed" / "paeruginosa_features_combined.csv"
SNP_FEATURES_OUT = PROJECT_ROOT / "data" / "processed" / "paeruginosa_snp_features.csv"

REFERENCE_GBFF = PROJECT_ROOT / "data" / "raw" / "references" / "GCF_000006765.1_ASM676v1_genomic.gbff"
SNIPPY_ENV = "pathtwin-snippy"

# gene -> the symbol actually used in PAO1's RefSeq GenBank annotation.
# mexZ is annotated there as "nfxB" (its older/alternate name) -- verified
# directly against the .gbff file, not assumed from memory.
TARGET_GENES = {
    "mexR": "mexR",
    "nalC": "nalC",
    "nalD": "nalD",
    "mexZ": "nfxB",
    "oprD": "oprD",
}

LOF_EFFECTS = {"frameshift_variant", "stop_gained", "start_lost"}
# Any coding-change effect token that counts as a (non-LOF) missense-class
# feature when present without an LOF token.
MISSENSE_EFFECTS = {
    "missense_variant", "conservative_inframe_deletion",
    "conservative_inframe_insertion", "disruptive_inframe_deletion",
    "disruptive_inframe_insertion", "initiator_codon_variant",
}


def run_rgi(genome_id: str, fasta: Path, force: bool = False) -> dict:
    outdir = RGI_WORKDIR / genome_id
    outdir.mkdir(parents=True, exist_ok=True)
    output_base = outdir / "rgi"
    output_txt = output_base.with_suffix(".txt")

    if not output_txt.exists() or force:
        cmd = [
            "rgi", "main",
            "--input_sequence", str(fasta),
            "--output_file", str(output_base),
            "--input_type", "contig",
            "--clean",
            "-n", "2",
            "-a", "DIAMOND",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0 or not output_txt.exists():
            print(f"    RGI FAILED for {genome_id}: {result.stderr[-500:]}", file=sys.stderr)
            return {}

    genes = {}
    with open(output_txt) as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            genes[row["Best_Hit_ARO"]] = 1
    return genes


def run_snippy(genome_id: str, fasta: Path, force: bool = False) -> Path:
    outdir = SNIPPY_WORKDIR / genome_id
    snps_tab = outdir / "snps.tab"
    if snps_tab.exists() and not force:
        return snps_tab

    cmd = [
        "conda", "run", "-n", SNIPPY_ENV, "--no-capture-output",
        "snippy",
        "--outdir", str(outdir),
        "--ref", str(REFERENCE_GBFF),
        "--ctgs", str(fasta),
        "--cpus", "2",
        "--force",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if not snps_tab.exists():
        print(f"    snippy FAILED for {genome_id}: {result.stderr[-500:]}", file=sys.stderr)
        return None
    return snps_tab


def parse_snp_features(snps_tab: Path) -> dict:
    """Returns {gene_key}_missense / {gene_key}_lof binary flags for the
    5 target genes, from one genome's snps.tab."""
    features = {}
    for gene_key in TARGET_GENES:
        features[f"{gene_key}_missense"] = 0
        features[f"{gene_key}_lof"] = 0

    if snps_tab is None or not snps_tab.exists():
        return features

    symbol_to_key = {v: k for k, v in TARGET_GENES.items()}

    with open(snps_tab) as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            gene_symbol = row.get("GENE", "")
            if gene_symbol not in symbol_to_key:
                continue
            gene_key = symbol_to_key[gene_symbol]
            effect = row.get("EFFECT", "")
            tokens = set(effect.replace("&", " ").split())
            if tokens & LOF_EFFECTS:
                features[f"{gene_key}_lof"] = 1
            elif tokens & MISSENSE_EFFECTS:
                features[f"{gene_key}_missense"] = 1
    return features


def process_one(genome_id: str, fasta_str: str, force_rgi: bool, force_snippy: bool):
    """Runs in a worker process: RGI + snippy + SNP-feature parsing for one
    genome. Returns (genome_id, genes_dict_or_None, snp_feats_dict, error_or_None)."""
    fasta = Path(fasta_str)
    genes = run_rgi(genome_id, fasta, force=force_rgi)
    if not genes:
        return genome_id, None, None, "RGI failed"

    snps_tab = run_snippy(genome_id, fasta, force=force_snippy)
    snp_feats = parse_snp_features(snps_tab)
    error = "snippy failed" if snps_tab is None else None
    return genome_id, genes, snp_feats, error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None,
                         help="Process only the first N genomes (for testing)")
    parser.add_argument("--workers", type=int, default=6,
                         help="Number of genomes to process concurrently (default: 6)")
    parser.add_argument("--force-rgi", action="store_true")
    parser.add_argument("--force-snippy", action="store_true")
    args = parser.parse_args()

    if not METADATA_CSV.exists():
        raise SystemExit(f"Metadata not found: {METADATA_CSV}. Run prep_paeruginosa_bvbrc_data.py first.")

    meta = pd.read_csv(METADATA_CSV, dtype=str)
    if args.limit:
        meta = meta.head(args.limit)

    RGI_WORKDIR.mkdir(parents=True, exist_ok=True)
    SNIPPY_WORKDIR.mkdir(parents=True, exist_ok=True)

    rgi_rows = {}
    snp_rows = {}
    failures = []

    total = len(meta)
    tasks = []
    for row in meta.itertuples(index=False):
        genome_id = row.genome_id
        fasta = PROJECT_ROOT / row.assembly_path
        if not fasta.exists():
            failures.append((genome_id, "assembly missing"))
            continue
        tasks.append((genome_id, str(fasta)))

    print(f"Processing {len(tasks)} genomes with {args.workers} parallel workers "
          f"(each RGI/snippy call itself uses 4 threads)...", flush=True)

    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_one, gid, fpath, args.force_rgi, args.force_snippy): gid
            for gid, fpath in tasks
        }
        for future in as_completed(futures):
            genome_id = futures[future]
            done += 1
            try:
                gid, genes, snp_feats, error = future.result()
            except Exception as e:
                failures.append((genome_id, f"worker exception: {e}"))
                print(f"[{done}/{total}] {genome_id}: EXCEPTION {e}", flush=True)
                continue

            if genes is None:
                failures.append((gid, error or "unknown failure"))
                print(f"[{done}/{total}] {gid}: FAILED ({error})", flush=True)
                continue

            rgi_rows[gid] = genes
            snp_rows[gid] = snp_feats
            if error:
                failures.append((gid, error))
            if done % 10 == 0 or done == total:
                print(f"[{done}/{total}] completed (last: {gid})", flush=True)

    print(f"\nCompleted {len(rgi_rows)}/{total} genomes with usable RGI output "
          f"({len(snp_rows)} with SNP features attempted).")
    if failures:
        print(f"{len(failures)} failure(s):")
        for gid, reason in failures[:20]:
            print(f"  {gid}: {reason}")
        if len(failures) > 20:
            print(f"  ... and {len(failures) - 20} more")

    # Build baseline (gene presence/absence) matrix
    all_genes = sorted({g for genes in rgi_rows.values() for g in genes})
    label_map = dict(zip(meta["genome_id"], meta["is_resistant"].astype(int)))

    baseline_records = []
    for genome_id, genes in rgi_rows.items():
        rec = {"genome_id": genome_id, "is_resistant": label_map[genome_id]}
        for g in all_genes:
            rec[g] = genes.get(g, 0)
        baseline_records.append(rec)
    baseline_df = pd.DataFrame(baseline_records)
    baseline_df.to_csv(BASELINE_OUT, index=False)
    print(f"\nWrote baseline (presence-only) feature matrix: {BASELINE_OUT} "
          f"({len(baseline_df)} genomes x {len(all_genes)} gene features)")

    # SNP features on their own (useful for inspection / the combined build)
    snp_df = pd.DataFrame([{"genome_id": gid, **feats} for gid, feats in snp_rows.items()])
    snp_df.to_csv(SNP_FEATURES_OUT, index=False)
    print(f"Wrote SNP feature table: {SNP_FEATURES_OUT} ({len(snp_df)} genomes x "
          f"{len(snp_df.columns) - 1} SNP features)")

    # Combined matrix: baseline + SNP features, same genome set
    combined_df = baseline_df.merge(snp_df, on="genome_id", how="left")
    snp_cols = [c for c in snp_df.columns if c != "genome_id"]
    combined_df[snp_cols] = combined_df[snp_cols].fillna(0).astype(int)
    combined_df.to_csv(COMBINED_OUT, index=False)
    print(f"Wrote combined (presence + SNP) feature matrix: {COMBINED_OUT} "
          f"({len(combined_df)} genomes x {len(all_genes) + len(snp_cols)} features)")

    # Quick summary of SNP feature prevalence
    print("\nSNP feature prevalence (non-zero count / total genomes with SNP data attempted):")
    for col in snp_cols:
        n = int(snp_df[col].sum())
        print(f"  {col}: {n}/{len(snp_df)}")


if __name__ == "__main__":
    main()

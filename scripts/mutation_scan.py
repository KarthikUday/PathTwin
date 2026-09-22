#!/usr/bin/env python3
"""Stage 3: mutation scanning against a species' reference genome.

Looks up a species' reference genome accession from
config/reference_genomes.yml, locates the matching downloaded reference
FASTA under data/raw/references/, and runs snippy (contig mode) to call
SNPs/variants between a query genome assembly and that reference.

snippy lives in its own dedicated conda env ("pathtwin-snippy"), not in
"pathtwin" -- its bioconda package pulls in dependency pins (an ancient
samtools 0.1.19 build, for older snippy releases' compiled Perl bindings)
that conflict with rgi's and biopython's requirements in the shared
"pathtwin" env. This script shells out to it via `conda run -n
pathtwin-snippy`, so it works regardless of which env is currently active.

This performs raw variant calling only -- it does NOT filter for known
resistance-associated positions. That comes in a later stage.
"""
import argparse
import subprocess
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "reference_genomes.yml"
REFERENCE_DIR = PROJECT_ROOT / "data" / "raw" / "references"
SNIPPY_ENV = "pathtwin-snippy"


def load_reference(species: str) -> dict:
    with open(CONFIG_PATH) as fh:
        config = yaml.safe_load(fh)

    references = (config or {}).get("references", {})
    if species not in references:
        available = ", ".join(sorted(references)) or "(none configured)"
        raise SystemExit(f"Unknown species '{species}'. Available: {available}")
    return references[species]


def find_reference_file(accession: str) -> Path:
    # Prefer an annotated GenBank flat file (<accession>_<assembly>_genomic.gbff)
    # over a bare FASTA (<accession>_<assembly>_genomic.fna): snippy only emits
    # gene/amino-acid-level SnpEff annotations (GENE, AA_POS, EFFECT columns in
    # snps.tab) when given a reference that carries gene annotations. A bare
    # FASTA yields nucleotide-only calls, which resistance-SNP filtering can't
    # use since CARD's curated SNP positions are amino-acid positions within a
    # gene's protein, not raw genomic coordinates.
    gbff_matches = sorted(REFERENCE_DIR.glob(f"{accession}_*_genomic.gbff"))
    if gbff_matches:
        if len(gbff_matches) > 1:
            raise SystemExit(
                f"Multiple candidate GenBank references for '{accession}' found: "
                f"{[m.name for m in gbff_matches]}. Remove the stale one(s)."
            )
        return gbff_matches[0]

    fna_matches = sorted(REFERENCE_DIR.glob(f"{accession}_*_genomic.fna"))
    if not fna_matches:
        raise SystemExit(
            f"No reference file found for accession '{accession}' in "
            f"{REFERENCE_DIR}. Expected '{accession}_<assembly_name>_genomic.gbff' "
            f"(preferred, for gene-level annotation) or '..._genomic.fna'."
        )
    if len(fna_matches) > 1:
        raise SystemExit(
            f"Multiple candidate reference FASTAs for '{accession}' found: "
            f"{[m.name for m in fna_matches]}. Remove the stale one(s)."
        )
    print(
        "WARNING: no .gbff found, falling back to a bare FASTA reference -- "
        "variants will lack gene/amino-acid annotation, so resistance-SNP "
        "filtering (scripts/filter_resistance_snps.py) will find nothing.",
        file=sys.stderr,
    )
    return fna_matches[0]


def run_snippy(reference: Path, query_genome: Path, outdir: Path, cpus: int) -> None:
    outdir.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "conda", "run", "-n", SNIPPY_ENV, "--no-capture-output",
        "snippy",
        "--outdir", str(outdir),
        "--ref", str(reference),
        "--ctgs", str(query_genome),
        "--cpus", str(cpus),
        "--force",
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run snippy variant calling for a query genome assembly against "
            "its species' reference genome (raw SNP/variant calling only -- "
            "no resistance-position filtering)."
        )
    )
    parser.add_argument(
        "--species",
        required=True,
        help="Species key from config/reference_genomes.yml (e.g. klebsiella_pneumoniae)",
    )
    parser.add_argument(
        "--query-genome",
        required=True,
        type=Path,
        help="Path to the query genome assembly FASTA to call variants for",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help=(
            "Output directory for snippy results "
            "(default: results/<query-genome-stem>_mutations/)"
        ),
    )
    parser.add_argument(
        "--cpus",
        type=int,
        default=4,
        help="Number of CPUs for snippy to use (default: 4)",
    )
    args = parser.parse_args()

    if not args.query_genome.exists():
        raise SystemExit(f"Query genome not found: {args.query_genome}")

    ref_info = load_reference(args.species)
    reference_fasta = find_reference_file(ref_info["accession"])

    outdir = args.outdir or (
        PROJECT_ROOT / "results" / f"{args.query_genome.stem}_mutations"
    )

    print(f"Species:          {args.species} ({ref_info.get('display_name', '')})")
    print(f"Reference:        {reference_fasta}")
    print(f"Query genome:     {args.query_genome}")
    print(f"Output directory: {outdir}")

    run_snippy(reference_fasta, args.query_genome, outdir, args.cpus)

    print("Done.")


if __name__ == "__main__":
    main()

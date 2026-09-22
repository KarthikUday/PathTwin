#!/usr/bin/env python3
"""
PathTwin — Gene-presence feature matrix builder (ML pipeline, pre-Stage 5)

Runs the existing Stage 1 RGI resistance-gene profiling pipeline (see
README's "Stage 1" section and the Stage 1-adjacent conventions used by
scripts/mutation_scan.py) across a directory of genome assemblies and
assembles a gene presence/absence feature matrix suitable for training a
classifier (scripts/train_classifier.py).

Each genome contributes a 1 for every CARD gene that RGI actually detects
(Best_Hit_ARO from rgi main's tabular output). Genes not detected in any
genome in the batch do not become columns, while a genome missing a gene
detected in another genome receives an implicit 0 when the matrix is
assembled. This produces a standard presence/absence matrix, similar to
roary: only observed columns are materialized, while absence is implicit.

Requires the `pathtwin` conda environment (RGI on PATH, with the CARD
database already loaded via `rgi auto_load`; see README's "CARD database
setup").

genome_to_features() includes a blaSHV sanity check for K. pneumoniae
genomes: blaSHV is intrinsic to the species (present in essentially every
K. pneumoniae genome, chromosomal rather than acquired), so an RGI result
with no blaSHV hit is a strong indication that the CARD database is
mis-loaded (see README's "CARD database setup" and "RGI/CARD database
mis-load recurrence" sections), rather than evidence that the gene is
absent. This check was introduced after a Stage 5 development run produced
only 6 total genes across 3 genomes, compared with ~36 genes detected in a
single genome.
"""

import argparse
import csv
import subprocess
import sys
import warnings
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RGI_WORKDIR = PROJECT_ROOT / "results" / "rgi_features"

GENOME_GLOBS = ("*.fna", "*.fasta", "*.fa")
_SPECIES_HEADER_SNIFF_LINES = 5


def _looks_like_klebsiella_pneumoniae(genome_fasta_path: Path) -> bool:
    """
    Best-effort species sniff from the genome FASTA's own header lines --
    NCBI assembly FASTAs carry the organism name on each contig's
    description line (e.g. '>NZ_CP040993.1 Klebsiella pneumoniae strain
    FDAARGOS_775 chromosome...'). Used only to decide whether the blaSHV
    sanity check below applies; never raises, just returns False if it
    can't tell (a header in an unexpected format is not itself suspicious).
    """
    try:
        with open(genome_fasta_path) as f:
            for _ in range(_SPECIES_HEADER_SNIFF_LINES):
                line = f.readline()
                if not line:
                    break
                if line.startswith(">") and "klebsiella pneumoniae" in line.lower():
                    return True
    except OSError:
        pass
    return False


def genome_to_features(genome_fasta_path, force_rerun: bool = False) -> dict:
    """
    Run RGI's Stage 1 pipeline (`rgi main`) on a single genome assembly and
    return its gene-presence feature vector.

    Returns a dict {gene_name: 1} for every gene RGI detected (Best_Hit_ARO
    in its tabular output) -- one entry per unique hit. Genes RGI did not
    detect in this genome are simply absent from the dict; callers that
    need an explicit 0 for "not present" (e.g. build_feature_matrix, when
    assembling a matrix across genomes with different gene sets) should
    treat any key missing from this dict as 0.

    `force_rerun=True` re-runs RGI even if cached output from a previous
    call already exists for this genome (default: reuse the cache -- RGI's
    BLAST/DIAMOND search against CARD is the expensive step, and its
    output is deterministic for a given genome + CARD database version).
    """
    genome_fasta_path = Path(genome_fasta_path)
    if not genome_fasta_path.exists():
        raise FileNotFoundError(f"Genome FASTA not found: {genome_fasta_path}")

    genome_id = genome_fasta_path.stem
    outdir = RGI_WORKDIR / genome_id
    outdir.mkdir(parents=True, exist_ok=True)
    output_base = outdir / "rgi"
    output_txt = output_base.with_suffix(".txt")

    if force_rerun or not output_txt.exists():
        cmd = [
            "rgi", "main",
            "--input_sequence", str(genome_fasta_path),
            "--output_file", str(output_base),
            "--input_type", "contig",
            "--clean",
        ]
        print(f"  Running RGI on {genome_fasta_path.name}...")
        result = subprocess.run(cmd, cwd=outdir, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"RGI failed on {genome_fasta_path}:\n"
                f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )
        if not output_txt.exists():
            raise FileNotFoundError(
                f"RGI reported success but expected output not found: {output_txt}\n"
                f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )
    else:
        print(f"  Reusing cached RGI output for {genome_fasta_path.name} ({output_txt})")

    features = {}
    with open(output_txt, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames is None or "Best_Hit_ARO" not in reader.fieldnames:
            raise ValueError(
                f"RGI output at {output_txt} is missing the expected "
                f"'Best_Hit_ARO' column -- got columns: {reader.fieldnames}"
            )
        for row in reader:
            gene = row["Best_Hit_ARO"].strip()
            if gene:
                features[gene] = 1

    # Sanity check, not a silent pass: blaSHV (CARD reports it under the
    # specific allele name, e.g. "SHV-1"/"SHV-11" -- match on "SHV"
    # generically) is intrinsic to K. pneumoniae. Finding none in a genome
    # that is K. pneumoniae is a strong signal the CARD database is
    # mis-loaded (silently dropped hits -- exactly what happened before;
    # see README), not that this particular genome genuinely lacks it.
    if _looks_like_klebsiella_pneumoniae(genome_fasta_path):
        if not any("SHV" in gene.upper() for gene in features):
            warnings.warn(
                f"{genome_fasta_path.name} looks like Klebsiella pneumoniae "
                f"(per its FASTA header) but RGI found no SHV beta-lactamase "
                f"(blaSHV) hit. blaSHV is intrinsic to this species -- its "
                f"absence almost certainly means the CARD database in this "
                f"env is mis-loaded (silently-dropped hits), not that this "
                f"genome genuinely lacks it. Re-run `rgi auto_load --clean` "
                f"(see README's 'CARD database setup') and re-verify before "
                f"trusting this result.",
                RuntimeWarning,
                stacklevel=2,
            )

    return features


def build_feature_matrix(genome_dir, output_csv) -> pd.DataFrame:
    """
    Run genome_to_features across every genome assembly in genome_dir,
    assemble a gene presence/absence matrix (rows = genome_id, columns =
    every gene detected in at least one genome in the batch, 1/0 values),
    and write it to output_csv.
    """
    genome_dir = Path(genome_dir)
    if not genome_dir.is_dir():
        raise NotADirectoryError(f"Genome directory not found: {genome_dir}")

    genome_files = sorted(
        {p for pattern in GENOME_GLOBS for p in genome_dir.glob(pattern)}
    )
    if not genome_files:
        raise FileNotFoundError(
            f"No genome FASTA files found in {genome_dir} "
            f"(looked for {', '.join(GENOME_GLOBS)})"
        )

    per_genome_features = {}
    for genome_fasta in genome_files:
        genome_id = genome_fasta.stem
        print(f"Processing {genome_id}...")
        per_genome_features[genome_id] = genome_to_features(genome_fasta)

    matrix = pd.DataFrame.from_dict(per_genome_features, orient="index")
    matrix = matrix.fillna(0).astype(int)
    matrix = matrix.sort_index().sort_index(axis=1)
    matrix.index.name = "genome_id"

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(output_csv)

    print(
        f"\nFeature matrix: {matrix.shape[0]} genomes x {matrix.shape[1]} genes "
        f"-> {output_csv}"
    )
    return matrix


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build a gene presence/absence feature matrix across a directory "
            "of genome assemblies, via RGI (Stage 1)."
        )
    )
    parser.add_argument(
        "--genome-dir", required=True,
        help="Directory containing genome assembly FASTA files (.fna/.fasta/.fa)",
    )
    parser.add_argument(
        "--output-csv", required=True,
        help="Path to write the resulting feature matrix CSV",
    )
    args = parser.parse_args()

    try:
        build_feature_matrix(args.genome_dir, args.output_csv)
    except Exception as e:
        sys.exit(f"ERROR: {e}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Stage 3: filter snippy's raw variants down to CARD resistance-conferring SNPs.

snippy's snps.tab gives, per variant, a genomic (CHROM, POS) location and --
for CDS-overlapping variants -- a gene-relative amino-acid change (e.g.
"p.Ser83Leu" in gene gyrA), produced by the snpEff annotation step that runs
automatically when snippy is given a GenBank-format reference (see
mutation_scan.py). CARD's curated resistance SNPs (from the "protein variant
model" and "protein overexpression model" entries in card.json) are recorded
the same way: a gene, an amino-acid position, and a specific wild-type ->
mutant residue change (e.g. "S83L"). Both datasets describe amino-acid-level
changes, so they can be joined on (gene, AA position, mutant residue) --
never on raw genomic coordinates, which differ between any two arbitrary
reference genomes.

MATCHING CRITERION -- deliberately strict, to avoid false positives from
gene-numbering drift between CARD's own curated reference protein for a
model and our reference genome's copy of that gene:
  1. gene name matches (case-insensitive)
  2. amino-acid position matches
  3. the wild-type residue snippy calls at that position (i.e. what our
     reference genome carries) matches CARD's curated wild-type residue --
     this is a sanity check that the position numbering is actually
     consistent between the two protein sequences, not just coincidentally
     the same integer
  4. the mutant residue our query genome carries at that position matches
     CARD's curated resistant mutant residue exactly

Only variants passing all four are "resistance-relevant" here. A variant
landing on a curated position but changing to a *different* residue than
CARD's curated resistant allele is NOT included -- CARD's curated SNPs are
specific, validated substitutions (e.g. gyrA S83L is a known fluoroquinolone
resistance mutation; gyrA S83A is not necessarily one), so a same-position,
different-residue change is not evidence of the same resistance phenotype.
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CARD_JSON = Path.home() / "card-data" / "card.json"

RELEVANT_MODEL_TYPES = {"protein variant model", "protein overexpression model"}

# Standard 3-letter -> 1-letter amino acid codes, as used in snpEff's HGVS.p
# notation (e.g. "p.Ser83Leu"). "Ter" is a stop codon.
AA_3TO1 = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C",
    "Gln": "Q", "Glu": "E", "Gly": "G", "His": "H", "Ile": "I",
    "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F", "Pro": "P",
    "Ser": "S", "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V",
    "Ter": "*", "Sec": "U", "Pyl": "O",
}

CARD_SNP_RE = re.compile(r"^([A-Za-z*])(\d+)([A-Za-z*])$")
HGVS_P_RE = re.compile(r"p\.([A-Za-z]{3})(\d+)([A-Za-z]{3})")

# Strips a leading "Genus species[ subsp. name]" prefix from a CARD model
# name, e.g. "Escherichia coli " or "Klebsiella pneumoniae subsp. pneumoniae ".
_SPECIES_PREFIX_RE = re.compile(
    r"^[A-Z][a-z]+\s+[a-z]+(?:\s+subsp\.\s+[a-z]+)?\s+"
)
# The phrase that separates "<gene>" from the resistance description in a
# CARD model name, e.g. "... EF-Tu [mutants conferring] resistance to ...",
# "... MarR [with mutation] associated with ...", "... PBP1a [conferring]
# resistance to ...".
_TRIGGER_RE = re.compile(
    r"\s+(?:conferring|confers?|decreased\s+susceptibility|"
    r"with\s+mutations?|mutants?\s+conferring|mutations?\s+conferring|"
    r"associated\s+with)\b",
    re.IGNORECASE,
)


def extract_gene_symbol(model_name: str) -> str:
    """Best-effort extraction of a bare gene symbol from a CARD model name.

    Many CARD protein variant models are already a bare gene symbol (e.g.
    "basR", "lmrA"). Others are a full descriptive phrase (e.g. "Escherichia
    coli EF-Tu mutants conferring resistance to Pulvomycin") -- for those we
    strip the leading species binomial and trailing resistance-description
    clause to isolate the gene token(s) in between.
    """
    if " " not in model_name:
        return model_name
    trigger = _TRIGGER_RE.search(model_name)
    prefix = model_name[: trigger.start()] if trigger else model_name
    prefix = _SPECIES_PREFIX_RE.sub("", prefix, count=1).strip()
    return prefix or model_name


def load_card_curated_snps(card_json_path: Path) -> dict:
    """Build a lookup of CARD's curated resistance SNPs.

    Returns {(gene_lower, position): [record, ...]} where each record has
    gene, position, wildtype, mutant, aro_accession, aro_id, aro_name,
    model_name, model_type, drug_classes.
    """
    with open(card_json_path) as fh:
        card = json.load(fh)

    lookup: dict = {}
    n_models = 0
    n_snps = 0

    for key, entry in card.items():
        if not isinstance(entry, dict) or entry.get("model_type") not in RELEVANT_MODEL_TYPES:
            continue
        snp_param = entry.get("model_param", {}).get("snp", {})
        param_value = snp_param.get("param_value")
        if not param_value:
            continue

        n_models += 1
        gene = extract_gene_symbol(entry.get("model_name", ""))
        drug_classes = sorted(
            {
                cat["category_aro_name"]
                for cat in entry.get("ARO_category", {}).values()
                if cat.get("category_aro_class_name") == "Drug Class"
            }
        )

        for raw_snp in param_value.values():
            m = CARD_SNP_RE.match(raw_snp.strip())
            if not m:
                continue
            wildtype, position_str, mutant = m.groups()
            position = int(position_str)
            record = {
                "gene": gene,
                "position": position,
                "wildtype": wildtype.upper(),
                "mutant": mutant.upper(),
                "curated_change": raw_snp.strip(),
                "aro_accession": entry.get("ARO_accession", ""),
                "aro_id": entry.get("ARO_id", ""),
                "aro_name": entry.get("ARO_name", ""),
                "model_name": entry.get("model_name", ""),
                "model_type": entry.get("model_type", ""),
                "drug_classes": drug_classes,
            }
            lookup.setdefault((gene.lower(), position), []).append(record)
            n_snps += 1

    print(
        f"Loaded {n_snps} curated resistance SNPs from {n_models} CARD "
        f"protein variant/overexpression models.",
        file=sys.stderr,
    )
    return lookup


def parse_snippy_tab(tab_path: Path):
    """Yield one dict per snippy snps.tab row with a parseable HGVS.p change."""
    with open(tab_path, newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            effect = row.get("EFFECT", "")
            m = HGVS_P_RE.search(effect)
            if not m:
                continue
            wt3, pos_str, mut3 = m.groups()
            wildtype = AA_3TO1.get(wt3)
            mutant = AA_3TO1.get(mut3)
            if wildtype is None or mutant is None or wildtype == mutant:
                continue  # unmappable code, or synonymous (no actual AA change)
            row["_aa_wildtype"] = wildtype
            row["_aa_position"] = int(pos_str)
            row["_aa_mutant"] = mutant
            yield row


def find_resistance_relevant_snps(tab_path: Path, card_lookup: dict) -> list:
    results = []
    for row in parse_snippy_tab(tab_path):
        gene = row.get("GENE", "")
        if not gene:
            continue
        candidates = card_lookup.get((gene.lower(), row["_aa_position"]))
        if not candidates:
            continue
        for card_record in candidates:
            if card_record["wildtype"] != row["_aa_wildtype"]:
                # Numbering doesn't line up with CARD's curated reference
                # protein at this position -- not a trustworthy match.
                continue
            if card_record["mutant"] != row["_aa_mutant"]:
                # Same curated position, but a different substitution than
                # CARD's specific validated resistance mutation.
                continue
            results.append(
                {
                    "gene": gene,
                    "locus_tag": row.get("LOCUS_TAG", ""),
                    "chrom": row.get("CHROM", ""),
                    "genomic_pos": row.get("POS", ""),
                    "ref_allele": row.get("REF", ""),
                    "alt_allele": row.get("ALT", ""),
                    "aa_change": f"{card_record['wildtype']}{card_record['position']}{card_record['mutant']}",
                    "drug_class": "; ".join(card_record["drug_classes"]) or "n/a",
                    "aro_accession": card_record["aro_accession"],
                    "aro_id": card_record["aro_id"],
                    "aro_name": card_record["aro_name"],
                    "card_model_type": card_record["model_type"],
                }
            )
    return results


def write_output_table(rows: list, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "gene", "locus_tag", "chrom", "genomic_pos", "ref_allele", "alt_allele",
        "aa_change", "drug_class", "aro_accession", "aro_id", "aro_name",
        "card_model_type",
    ]
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Cross-reference snippy's raw variant calls against CARD's curated "
            "resistance-conferring SNP positions, keeping only variants that "
            "exactly match a known resistance mutation (gene + AA position + "
            "specific mutant residue)."
        )
    )
    parser.add_argument(
        "--snippy-dir",
        required=True,
        type=Path,
        help="snippy output directory (containing snps.tab)",
    )
    parser.add_argument(
        "--card-json",
        type=Path,
        default=DEFAULT_CARD_JSON,
        help=f"Path to CARD's card.json (default: {DEFAULT_CARD_JSON})",
    )
    parser.add_argument(
        "--outfile",
        type=Path,
        default=None,
        help="Output path (default: <snippy-dir>/resistance_relevant_snps.tab)",
    )
    args = parser.parse_args()

    tab_path = args.snippy_dir / "snps.tab"
    if not tab_path.exists():
        raise SystemExit(f"snps.tab not found: {tab_path}")
    if not args.card_json.exists():
        raise SystemExit(f"card.json not found: {args.card_json}")

    card_lookup = load_card_curated_snps(args.card_json)
    matches = find_resistance_relevant_snps(tab_path, card_lookup)

    outfile = args.outfile or (args.snippy_dir / "resistance_relevant_snps.tab")
    write_output_table(matches, outfile)

    total_variants = sum(1 for _ in open(tab_path)) - 1  # minus header
    print(f"Raw variants scanned:        {total_variants}")
    print(f"Resistance-relevant matches: {len(matches)}")
    print(f"Written to:                  {outfile}")
    for m in matches:
        print(
            f"  {m['gene']} {m['aa_change']}  ({m['drug_class']})  "
            f"{m['aro_name']} [ARO:{m['aro_accession']}]"
        )


if __name__ == "__main__":
    main()

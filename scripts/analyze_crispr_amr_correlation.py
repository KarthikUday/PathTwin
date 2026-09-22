#!/usr/bin/env python3
"""
PathTwin Stage 6 v2 -- CRISPR-Cas / acquired-resistance co-occurrence at
scale (1020 P. aeruginosa genomes from Stage 5's fluoroquinolone classifier
work; 400 newly-downloaded S. aureus genomes, stratified-sampled from
Stage 5's 1,475-genome oxacillin-labeled pool).

Reuses Stage 1's intrinsic-vs-acquired convention (documented in README's
Stage 1 / Stage 6 sections): chromosomal, species-defining determinants
(efflux pumps and their regulators, intrinsic beta-lactamases, porins,
target-site-mutation flags, weak van*-cluster fragments) are never counted
as "acquired," regardless of RGI/CARD hit confidence.

At this scale, classification is done programmatically against CARD's own
"AMR Gene Family" metadata (card.json) rather than re-deriving it gene by
gene, WITH one verified correction: CARD's family alone is not sufficient
for naturally single-copy intrinsic families (PDC beta-lactamase,
OXA-50-like beta-lactamase, fosfomycin thiol transferase, MATE transporter,
pmr phosphoethanolamine transferase) directly checked against PAO1 vs
PA14's own RGI output and confirmed PA14 carries BOTH the same single
intrinsic OXA-50-like gene PAO1 has (OXA-50) AND a second, distinct
OXA-50-like-family gene PAO1 lacks (OXA-488) CARD's original Stage 6
proof-of-concept called this second copy "acquired" for exactly this
reason, not because the family itself is mobile. Generalized here as: for
these single-copy-expected families, only genes BEYOND the first present
family member per genome count toward the acquired burden.
"""
import json
from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd
from scipy import stats

PROJECT_ROOT = Path("/home/kell/PathTwin")
CARD_JSON = Path("/home/kell/card-data/card.json")
CRISPR_DIR = PROJECT_ROOT / "results" / "crispr_scan"
OUT_DIR = CRISPR_DIR

# ---------------------------------------------------------------------
# Species-prefixed CARD entries that represent P. aeruginosa's own
# constitutive/core chromosomal genes (protein homolog model, always
# present, no mutation needed) -- same "named core gene" pattern already
# used for catB7/emrE/CpxR in the original 15-genome crossref.
PA_INTRINSIC_NAMED_GENES = {
    "Pseudomonas aeruginosa CpxR",
    "Pseudomonas aeruginosa catB7",
    "Pseudomonas aeruginosa emrE",
    "Pseudomonas aeruginosa soxR",
}
# Target-site-mutation flags and non-functional cluster fragments -- per
# README's established convention, never counted as acquired regardless
# of family.
PA_EXCLUDED_FLAGS_SUBSTR = [
    "with mutation conferring resistance",
    "conferring resistance to fluoroquinolones",  # gyrA/parE variant-model hits
    "vanW gene in van",  # weak cryptic glycopeptide cluster fragment
]

PA_INTRINSIC_FAMILIES = {
    "PDC beta-lactamase",
    "OXA-50-like beta-lactamase",
    "resistance-nodulation-cell division (RND) antibiotic efflux pump",
    "antibiotic efflux regulatory protein",
    "Outer Membrane Porin (Opr)",
    "pmr phosphoethanolamine transferase",
    "multidrug and toxic compound extrusion (MATE) transporter",
    "fosfomycin thiol transferase",
}
# Families expected to be a SINGLE intrinsic copy per genome -- extra
# distinct members beyond the first are acquired-like (the PA14
# OXA-50/OXA-488 precedent, verified directly above).
PA_SINGLE_COPY_INTRINSIC_FAMILIES = {
    "PDC beta-lactamase",
    "OXA-50-like beta-lactamase",
    "fosfomycin thiol transferase",
    "multidrug and toxic compound extrusion (MATE) transporter",
    "pmr phosphoethanolamine transferase",
}


def load_card_family_map():
    card = json.loads(CARD_JSON.read_text())
    name_to_families = {}
    for v in card.values():
        if not isinstance(v, dict):
            continue
        name = v.get("ARO_name", "")
        if name in name_to_families:
            continue
        cats = v.get("ARO_category", {})
        fams = [c.get("category_aro_name") for c in cats.values()
                if c.get("category_aro_class_name") == "AMR Gene Family"]
        name_to_families[name] = fams
    return name_to_families


def classify_pa_genes(feature_cols):
    """Returns {gene_name: 'intrinsic'|'acquired'}."""
    fam_map = load_card_family_map()
    classification = {}
    unmatched = []
    for g in feature_cols:
        if g in PA_INTRINSIC_NAMED_GENES:
            classification[g] = "intrinsic"
            continue
        if any(s in g for s in PA_EXCLUDED_FLAGS_SUBSTR):
            classification[g] = "excluded_flag"
            continue
        fams = fam_map.get(g)
        if fams is None:
            unmatched.append(g)
            classification[g] = "acquired"  # conservative default; reported below
            continue
        if set(fams) & PA_INTRINSIC_FAMILIES:
            classification[g] = "intrinsic"
        else:
            classification[g] = "acquired"
    return classification, unmatched, fam_map


def compute_pa_acquired_burden(features_df, classification, fam_map):
    """Per-genome acquired_gene_count, applying the single-copy-intrinsic-
    family extra-copy rule."""
    gene_cols = [c for c in features_df.columns if c not in ("genome_id", "is_resistant")]
    acquired_cols = [c for c in gene_cols if classification[c] == "acquired"]

    # group intrinsic single-copy-family genes by family
    family_members = defaultdict(list)
    for c in gene_cols:
        if classification[c] != "intrinsic":
            continue
        fams = set(fam_map.get(c, [])) & PA_SINGLE_COPY_INTRINSIC_FAMILIES
        for f in fams:
            family_members[f].append(c)

    burden = features_df[acquired_cols].sum(axis=1).astype(int) if acquired_cols else pd.Series(0, index=features_df.index)

    extra_copy_burden = pd.Series(0, index=features_df.index)
    for fam, members in family_members.items():
        present_count = features_df[members].sum(axis=1)
        extra_copy_burden += (present_count - 1).clip(lower=0)

    total_burden = burden + extra_copy_burden
    return total_burden, acquired_cols, dict(family_members)


def load_crispr_summaries(genome_ids, id_transform=lambda x: x):
    """Load results/crispr_scan/<genome_id>_summary.json for each genome_id
    (after applying id_transform to match the on-disk filename stem)."""
    rows = []
    missing = []
    for gid in genome_ids:
        fname = id_transform(gid)
        path = CRISPR_DIR / f"{fname}_summary.json"
        if not path.exists():
            missing.append(gid)
            continue
        d = json.loads(path.read_text())
        rows.append({
            "genome_id": gid,
            "cas_system_present": d["cas_system_present"],
            "cas_subtypes": tuple(d["cas_subtypes"]),
            "n_arrays_total": d["n_arrays_total"],
            "n_arrays_trusted": d["n_arrays_trusted"],
        })
    return pd.DataFrame(rows), missing


def mannwhitney_report(group_pos, group_neg, label):
    if len(group_pos) < 2 or len(group_neg) < 2:
        return {"label": label, "note": "insufficient n for test", "n_pos": len(group_pos), "n_neg": len(group_neg)}
    u, p = stats.mannwhitneyu(group_pos, group_neg, alternative="two-sided")
    return {
        "label": label,
        "n_pos": len(group_pos), "n_neg": len(group_neg),
        "median_pos": float(group_pos.median()), "median_neg": float(group_neg.median()),
        "mean_pos": float(group_pos.mean()), "mean_neg": float(group_neg.mean()),
        "mannwhitney_U": float(u), "p_value": float(p),
    }


def fisher_report(a_pos_b_pos, a_pos_b_neg, a_neg_b_pos, a_neg_b_neg, label):
    table = [[a_pos_b_pos, a_pos_b_neg], [a_neg_b_pos, a_neg_b_neg]]
    odds, p = stats.fisher_exact(table)
    return {"label": label, "table": table, "odds_ratio": float(odds), "p_value": float(p)}


# ============================== P. AERUGINOSA ==============================
def run_paeruginosa():
    print("=== P. aeruginosa ===")
    # dtype=str on genome_id: pandas' default float64 inference silently
    # collided distinct BV-BRC genome IDs (e.g. "287.12730" -> 287.1273)
    # and broke the crispr-summary filename lookup -- caught via an
    # impossible merged n (937) exceeding the loaded-summary count (921).
    features = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "paeruginosa_features_baseline.csv",
                            dtype={0: str})
    features = features.rename(columns={features.columns[0]: "genome_id"})
    gene_cols = [c for c in features.columns if c not in ("genome_id", "is_resistant")]
    classification, unmatched, fam_map = classify_pa_genes(gene_cols)
    print(f"  {len(gene_cols)} gene columns classified "
          f"({sum(1 for v in classification.values() if v=='intrinsic')} intrinsic, "
          f"{sum(1 for v in classification.values() if v=='acquired')} acquired, "
          f"{sum(1 for v in classification.values() if v=='excluded_flag')} excluded flags, "
          f"{len(unmatched)} unmatched-in-CARD [defaulted to acquired])")
    if unmatched:
        print(f"  UNMATCHED (not found in CARD by exact name, defaulted acquired): {unmatched}")

    burden, acquired_cols, single_copy_families = compute_pa_acquired_burden(features, classification, fam_map)
    features["acquired_gene_count"] = burden

    assert not features["genome_id"].duplicated().any(), \
        "duplicate genome_id in features matrix -- check for dtype-coercion collisions before trusting any stats"

    crispr_df, missing = load_crispr_summaries(features["genome_id"].tolist())
    print(f"  CRISPR-Cas summaries found for {len(crispr_df)}/{len(features)} genomes "
          f"({len(missing)} missing -- likely still running or failed)")

    merged = features.merge(crispr_df, on="genome_id", how="inner")
    n = len(merged)
    assert n <= len(crispr_df), \
        f"merged row count ({n}) exceeds loaded crispr-summary count ({len(crispr_df)}) -- merge fan-out bug"
    n_cas = int(merged["cas_system_present"].sum())
    frac_cas = n_cas / n if n else float("nan")
    print(f"  n={n}, CRISPR-Cas confidently present: {n_cas} ({frac_cas:.1%})")

    subtype_counter = Counter()
    for subtypes in merged["cas_subtypes"]:
        for s in subtypes:
            subtype_counter[s] += 1
    print(f"  Subtype distribution (genomes with >=1 confident operon of that subtype): {dict(subtype_counter)}")

    cas_pos = merged[merged["cas_system_present"]]
    cas_neg = merged[~merged["cas_system_present"]]

    mw_burden = mannwhitney_report(cas_pos["acquired_gene_count"], cas_neg["acquired_gene_count"],
                                    "acquired_gene_count: Cas-present vs Cas-absent")

    # binary acquired-burden (above-median) vs cas_present, Fisher
    median_burden = merged["acquired_gene_count"].median()
    high_burden = merged["acquired_gene_count"] > median_burden
    fisher_highburden = fisher_report(
        int((merged["cas_system_present"] & high_burden).sum()),
        int((merged["cas_system_present"] & ~high_burden).sum()),
        int((~merged["cas_system_present"] & high_burden).sum()),
        int((~merged["cas_system_present"] & ~high_burden).sum()),
        f"cas_present vs above-median acquired burden (median={median_burden})",
    )

    # cas_present vs the real fluoroquinolone-resistance phenotype label
    fisher_resistance = fisher_report(
        int((merged["cas_system_present"] & (merged["is_resistant"] == 1)).sum()),
        int((merged["cas_system_present"] & (merged["is_resistant"] == 0)).sum()),
        int((~merged["cas_system_present"] & (merged["is_resistant"] == 1)).sum()),
        int((~merged["cas_system_present"] & (merged["is_resistant"] == 0)).sum()),
        "cas_present vs fluoroquinolone-resistance phenotype",
    )

    # AYE-style check: does Cas-present correlate with HIGHER or LOWER
    # acquired burden -- direction, not just p-value
    direction = "HIGHER in Cas-present (contradicts literature)" if mw_burden.get("median_pos", 0) > mw_burden.get("median_neg", 0) else \
                ("LOWER in Cas-present (supports literature)" if mw_burden.get("median_pos", 1) < mw_burden.get("median_neg", 1) else "no difference")

    result = {
        "n_genomes_scanned": n,
        "n_cas_present": n_cas,
        "fraction_cas_present": frac_cas,
        "subtype_distribution": dict(subtype_counter),
        "acquired_vs_intrinsic_classification": classification,
        "unmatched_in_card_defaulted_acquired": unmatched,
        "single_copy_intrinsic_family_members_found": single_copy_families,
        "mannwhitney_acquired_burden_cas_present_vs_absent": mw_burden,
        "direction_vs_literature": direction,
        "fisher_cas_present_vs_high_acquired_burden": fisher_highburden,
        "fisher_cas_present_vs_fq_resistance_phenotype": fisher_resistance,
        "missing_crispr_summaries": missing,
    }
    merged.drop(columns=["cas_subtypes"]).to_csv(OUT_DIR / "paeruginosa_crispr_amr_merged.csv", index=False)
    (OUT_DIR / "paeruginosa_crispr_amr_stats.json").write_text(json.dumps(result, indent=2, default=str))
    print(f"  Saved: paeruginosa_crispr_amr_merged.csv, paeruginosa_crispr_amr_stats.json")
    return result


# ============================== S. AUREUS ==============================
def run_saureus():
    print("\n=== S. aureus ===")
    sample_labels = json.loads((PROJECT_ROOT / "data/raw/bvbrc_saureus/crispr_sample_labels.json").read_text())
    genome_ids = list(sample_labels.keys())

    crispr_df, missing = load_crispr_summaries(genome_ids)
    print(f"  CRISPR-Cas summaries found for {len(crispr_df)}/{len(genome_ids)} genomes "
          f"({len(missing)} missing -- likely still running or failed)")

    crispr_df["is_resistant"] = crispr_df["genome_id"].map(sample_labels)
    n = len(crispr_df)
    n_cas = int(crispr_df["cas_system_present"].sum())
    frac_cas = n_cas / n if n else float("nan")
    print(f"  n={n}, CRISPR-Cas confidently present: {n_cas} ({frac_cas:.1%})")

    subtype_counter = Counter()
    for subtypes in crispr_df["cas_subtypes"]:
        for s in subtypes:
            subtype_counter[s] += 1
    print(f"  Subtype distribution: {dict(subtype_counter)}")

    result = {
        "n_genomes_scanned": n,
        "n_cas_present": n_cas,
        "fraction_cas_present": frac_cas,
        "subtype_distribution": dict(subtype_counter),
        "missing_crispr_summaries": missing,
    }

    if n_cas >= 2:
        fisher_resistance = fisher_report(
            int((crispr_df["cas_system_present"] & (crispr_df["is_resistant"] == 1)).sum()),
            int((crispr_df["cas_system_present"] & (crispr_df["is_resistant"] == 0)).sum()),
            int((~crispr_df["cas_system_present"] & (crispr_df["is_resistant"] == 1)).sum()),
            int((~crispr_df["cas_system_present"] & (crispr_df["is_resistant"] == 0)).sum()),
            "cas_present vs oxacillin-resistance phenotype",
        )
        result["fisher_cas_present_vs_oxacillin_resistance"] = fisher_resistance
    else:
        result["fisher_cas_present_vs_oxacillin_resistance"] = (
            f"NOT COMPUTED -- only {n_cas} Cas-present genome(s) found; "
            f"a 2x2 test with a near-zero cell is not statistically meaningful"
        )

    crispr_df.to_csv(OUT_DIR / "saureus_crispr_amr_merged.csv", index=False)
    (OUT_DIR / "saureus_crispr_amr_stats.json").write_text(json.dumps(result, indent=2, default=str))
    print(f"  Saved: saureus_crispr_amr_merged.csv, saureus_crispr_amr_stats.json")
    return result


# ============== A. BAUMANNII / E. FAECIUM / E. CLOACAE (Stage 6 v3) ==============
# These three species' Stage 5 feature matrices come from BV-BRC's sp_gene
# collection (BLAT hits against CARD/NDARO/PATRIC, feature-keyed by gene
# symbol when present, else a verbose "description => gene" product
# string) -- NOT from a fresh RGI run like P. aeruginosa's baseline matrix.
# That naming style breaks exact-name CARD lookup badly (400+/634-670
# columns "unmatched" per species on a first pass, vs. 0 for P. aeruginosa)
# because many columns are BV-BRC's own pooled "family" rollup labels
# (e.g. "ADC family", "ACT/MIR family") rather than a single real CARD
# ARO entry, and many others are compound "EC description => GENE" text.
# Classification here is therefore keyword/regex-based on the visible
# column text, cross-checked directly against this project's own real
# Stage 1 RGI output (not just CARD's family metadata) for every
# borderline call before trusting it -- see the per-pattern comments below
# for exactly which real genome/hit verified which rule.
import re

# Verified directly against Stage 1's real RGI output before use:
#   A. baumannii ATCC19606 (type): single OXA-51-like hit (OXA-98,
#     Perfect, 100%) -- "OXA-51" appearing elsewhere in that same row was
#     only the AMR-Gene-Family description text, not a second hit.
#   ACICU: OXA-66 (OXA-51-like, intrinsic) + OXA-20, a genuinely SEPARATE
#     acquired gene under a different CARD family ("OXA beta-lactamase
#     without subfamily") -- confirms OXA-51-like does NOT need the
#     P. aeruginosa/OXA-50-like extra-copy correction: the family split
#     from other OXA families is already clean here.
#   AYE: OXA-69 (OXA-51-like, intrinsic) + OXA-10 (OXA-10-like, acquired)
#     -- same clean single-intrinsic-copy pattern.
#   ADC family: ADC-158 (ATCC19606), ADC-175 (ACICU), ADC-11 (AYE) -- one
#     intrinsic copy each, both CARD sub-families ("...pending
#     classification for carbapenemase activity" / "...without
#     carbapenemase activity") are the same intrinsic AmpC locus.
#   E. cloacae ATCC13047 (type): single CMH-4 hit. 9553: CMH-20 (same
#     intrinsic family) PLUS a separate acquired DHA-1 (different CARD
#     family "DHA beta-lactamase") -- confirms CMH is genuinely distinct
#     from acquired DHA/CMY-family plasmid AmpC, matching this project's
#     existing Stage 1/6 convention (DHA-1 already documented as
#     "acquired AmpC, non-carbapenemase" for this exact strain).
#   E. faecium DSM20477 (type) and DO (diversity): identical intrinsic
#     signature in both -- efmA, efrA, AAC(6')-Ii -- plus, in both, a
#     "vanY gene in vanB cluster" fragment hit, the same non-functional
#     cryptic-homolog pattern already established project-wide (neither
#     genome has a real van cluster; the ligase gene itself, vanA/vanB,
#     is absent in both).

# Fragment / target-site-mutation-flag patterns -- checked FIRST, before
# acquired/intrinsic classification, and never counted toward burden
# either way (same established convention as the weak van*-cluster
# fragments and gyrA/parE/ampR "with mutation" flags elsewhere in this
# project). A bare accessory van gene (vanH/X/Y/R/S/T/W/Z) without its
# own ligase confers nothing on its own -- verified directly above (both
# real E. faecium genomes checked carry the fragment with no ligase).
EXCLUDED_PATTERNS = [
    re.compile(r"gene in van\w* cluster", re.IGNORECASE),
    re.compile(r"\bmutant\b.*conferring resistance", re.IGNORECASE),
    re.compile(r"with mutation conferring resistance", re.IGNORECASE),
    # "...-type" / bare "unclassified" columns (e.g. "VanB-type",
    # "VanC/E/L/N-type") are BV-BRC's classification of which van-cluster
    # TYPE an accessory regulator (VanS/VanR) belongs to, not a distinct
    # gene call themselves -- verified directly against the full real
    # column list (data/processed/efaecium_features.csv): every "-type"
    # column has a sibling compound entry reading "...histidine kinase
    # VanS => <same type label>" or "...response regulator VanR => ...",
    # confirming these are VanS/VanR subtype tags, not ligase hits.
    re.compile(r"-type$", re.IGNORECASE),
    re.compile(r"^unclassified$", re.IGNORECASE),
    # ACCESSORY van-cluster genes -- H (dehydrogenase), R/S (two-component
    # regulator), T (racemase), W/Z (accessory/teicoplanin-specific), X/Y
    # (dipeptidase/carboxypeptidase) -- matched anywhere in the column
    # text (not anchored), since BV-BRC often embeds the real gene name
    # mid-sentence ("Vancomycin (or other glycopeptides) histidine kinase
    # VanS => ..."). Deliberately does NOT match the real ligase genes
    # themselves (VanA/B/C/D/E/G/L/M/N, the actual resistance-conferring
    # enzymes, handled by the ACQUIRED pattern below) -- verified this
    # distinction explicitly against a first-draft `^Van[A-Z]` version
    # that matched both by mistake (would have silently zeroed out the
    # real VanA/VanB signal for every genome) and against "Vancomycin"
    # itself as a substring (starts with "Van"+"c", which could collide
    # with a naive fixed-offset check -- confirmed safe here since "c"
    # is not in the H/R/S/T/W/X/Y/Z accessory-letter set).
    re.compile(r"Van(H|R|S|T|W|X|Y|Z)", re.IGNORECASE),
]

# Well-established ACQUIRED (horizontally-mobile) resistance-determinant
# families -- checked after the exclusions above. Deliberately specific
# (named families/gene prefixes, not generic substrings) after the
# "ACT" ⊂ "factor" false-positive lesson from the Stage 5 ML extension.
ACQUIRED_PATTERNS = [
    re.compile(r"AAC\(|ANT\(|APH\("),                                    # aminoglycoside-modifying enzymes
    re.compile(r"\b(SHV|TEM|CTX-M|CARB|GES|VIM|IMP|NDM|PER|VEB|KPC|CMY|DHA|FONA|SFO|LAP|TLA|IMI)\b"),  # mobile beta-lactamases
    re.compile(r"OXA-(10|23|24|40|58|134|143|213|286)\b"),               # acquired OXA carbapenemase/oxacillinase families (NOT OXA-51)
    re.compile(r"OXA beta-lactamase without subfamily", re.IGNORECASE),
    re.compile(r"\bqnr", re.IGNORECASE),                                 # plasmid-mediated quinolone resistance
    re.compile(r"dihydrofolate reductase dfr|^dfr", re.IGNORECASE),      # trimethoprim
    re.compile(r"\bsul[123]\b|sulfonamide resistant sul", re.IGNORECASE),  # sulfonamide
    re.compile(r"tet\(|tetracycline-resistant ribosomal|^tetR", re.IGNORECASE),  # tetracycline
    re.compile(r"erm\(|Erm-like|^msr|^mph\(|^lnu\(|^lsa\(|OptrA|\bCfr\b", re.IGNORECASE),  # macrolide/lincosamide/streptogramin
    re.compile(r"chloramphenicol acetyltransferase|^cat[A-Z]\d*\b|floR|cmlA|cmlB|^cmx", re.IGNORECASE),  # phenicol
    # (?![a-z]): rejects a match that continues into more lowercase
    # letters -- without this, the plain English word "vancomycin"
    # itself (contains "vanC" + "omycin") falsely matched as the VanC
    # ligase gene, caught by testing this pattern against the real full
    # column list before trusting it, not just against a few examples.
    # A real digit/slash/space/end-of-string after the letter (VanC1,
    # VanA/I/Pt-type, "VanA " mid-sentence, bare "VanA") still matches.
    re.compile(r"Van(A|B|C|D|E|G|L|M|N)(?![a-z])", re.IGNORECASE),
    re.compile(r"^qac|quaternary ammonium", re.IGNORECASE),              # disinfectant/mobile-element-associated
    re.compile(r"mcr-\d", re.IGNORECASE),                                # mobile colistin resistance
    re.compile(r"^arr-\d"),                                              # rifampin ADP-ribosyltransferase
    re.compile(r"16S rRNA.*methyltransferase", re.IGNORECASE),           # pan-aminoglycoside resistance (ArmA/RmtB-type)
    re.compile(r"[Bb]leomycin resistan(t|ce)"),
    re.compile(r"\bmecA\b|\bmecC\b"),
]

SPECIES_CONFIG = {
    "abaumannii": {
        "intrinsic_named_genes": set(),
        "intrinsic_patterns": [
            re.compile(r"OXA-51\b"),   # verified single-copy intrinsic above (ATCC19606/ACICU/AYE)
            re.compile(r"\bADC\b"),    # verified single-copy intrinsic above (all 3 real genomes)
        ],
    },
    "efaecium": {
        # Verified directly against real RGI output (DSM20477 + DO):
        # present in both, single copy, no acquired counterpart competing
        # for the same family in this species' feature set.
        "intrinsic_named_genes": {"efmA", "efrA", "efrB", "AAC(6')-Ii"},
        "intrinsic_patterns": [],
    },
    "ecloacae": {
        "intrinsic_named_genes": set(),
        "intrinsic_patterns": [
            re.compile(r"\bACT\b"), re.compile(r"\bMIR\b"),  # verified intrinsic above (ATCC13047/9553); word-boundary avoids the "factor" false-positive class of bug
            re.compile(r"\bCMH\b"),                           # verified intrinsic above (both real genomes)
        ],
    },
}

# Generic intrinsic fallback: CARD-family membership in core efflux/
# regulatory/porin categories, for whatever's left unclassified after the
# acquired-pattern check above. Reused across all 3 species since the
# same core chromosomal multidrug-efflux machinery (RND systems and their
# regulators, plus MFS/ABC components not already caught as an acquired
# gene by name) recurs across Gram-negatives and Gram-positives alike.
GENERIC_INTRINSIC_CARD_FAMILIES = {
    "resistance-nodulation-cell division (RND) antibiotic efflux pump",
    "antibiotic efflux regulatory protein",
    "Outer Membrane Porin (Opr)",
    "major facilitator superfamily (MFS) antibiotic efflux pump",
    "ATP-binding cassette (ABC) antibiotic efflux pump",
}


def classify_bvbrc_features(feature_cols, species_tag, fam_map):
    """Returns {col: 'intrinsic'|'acquired'|'excluded_flag'|'excluded_unclear'}.

    Conservative by design: anything matching no recognized
    acquired-determinant pattern AND no intrinsic signal defaults to
    'excluded_unclear', NOT 'acquired' -- unlike the P. aeruginosa
    classifier (which had 0 unmatched columns from RGI's clean gene-symbol
    output and could safely default unmatched-in-CARD to acquired). Here,
    ~60-70% of columns are BV-BRC sp_gene rollup/compound-description
    text that includes real non-AMR housekeeping genes pulled in under
    the "Antibiotic Resistance" property tag (e.g. "Alanine racemase",
    "Adenylate cyclase") -- defaulting those to 'acquired' would inflate
    the burden metric with noise instead of a real signal.
    """
    cfg = SPECIES_CONFIG[species_tag]
    classification = {}
    for col in feature_cols:
        if col in cfg["intrinsic_named_genes"]:
            classification[col] = "intrinsic"
            continue
        if any(p.search(col) for p in EXCLUDED_PATTERNS):
            classification[col] = "excluded_flag"
            continue
        if any(p.search(col) for p in ACQUIRED_PATTERNS):
            classification[col] = "acquired"
            continue
        if any(p.search(col) for p in cfg["intrinsic_patterns"]):
            classification[col] = "intrinsic"
            continue
        candidate = col.split("=>")[-1].strip() if "=>" in col else col
        fams = set(fam_map.get(candidate, []))
        if not fams:
            for part in re.split(r"[/,]", candidate):
                fams |= set(fam_map.get(part.strip(), []) or [])
        if fams & GENERIC_INTRINSIC_CARD_FAMILIES:
            classification[col] = "intrinsic"
            continue
        classification[col] = "excluded_unclear"
    return classification


def run_bvbrc_species(species_tag, display_name, features_csv, sample_labels_json, resistance_label_name):
    print(f"\n=== {display_name} ===")
    fam_map = load_card_family_map()

    sample_labels = json.loads(Path(sample_labels_json).read_text())
    genome_ids = list(sample_labels.keys())

    features = pd.read_csv(features_csv, dtype={"strain": str})
    features = features.rename(columns={features.columns[0]: "genome_id"})
    feature_cols = [c for c in features.columns if c not in ("genome_id", "is_resistant")]

    classification = classify_bvbrc_features(feature_cols, species_tag, fam_map)
    counts = Counter(classification.values())
    print(f"  {len(feature_cols)} feature columns classified "
          f"({counts['intrinsic']} intrinsic, {counts['acquired']} acquired, "
          f"{counts['excluded_flag']} excluded flags/fragments, "
          f"{counts['excluded_unclear']} excluded as unclear/non-AMR)")

    acquired_cols = [c for c in feature_cols if classification[c] == "acquired"]
    features["acquired_gene_count"] = features[acquired_cols].sum(axis=1).astype(int) if acquired_cols else 0

    # Restrict to the genomes actually CRISPR-scanned (the downloaded
    # assembly subset), not the full Stage 5 labeled pool.
    features = features[features["genome_id"].isin(genome_ids)].copy()
    assert not features["genome_id"].duplicated().any(), \
        "duplicate genome_id in features matrix -- check for dtype-coercion collisions before trusting any stats"

    crispr_df, missing = load_crispr_summaries(features["genome_id"].tolist())
    print(f"  CRISPR-Cas summaries found for {len(crispr_df)}/{len(features)} genomes "
          f"({len(missing)} missing -- likely still running or failed)")

    merged = features.merge(crispr_df, on="genome_id", how="inner")
    n = len(merged)
    assert n <= len(crispr_df), \
        f"merged row count ({n}) exceeds loaded crispr-summary count ({len(crispr_df)}) -- merge fan-out bug"
    n_cas = int(merged["cas_system_present"].sum())
    frac_cas = n_cas / n if n else float("nan")
    print(f"  n={n}, CRISPR-Cas confidently present: {n_cas} ({frac_cas:.1%})")

    subtype_counter = Counter()
    for subtypes in merged["cas_subtypes"]:
        for s in subtypes:
            subtype_counter[s] += 1
    print(f"  Subtype distribution: {dict(subtype_counter)}")

    cas_pos = merged[merged["cas_system_present"]]
    cas_neg = merged[~merged["cas_system_present"]]

    result = {
        "n_genomes_scanned": n,
        "n_cas_present": n_cas,
        "fraction_cas_present": frac_cas,
        "subtype_distribution": dict(subtype_counter),
        "acquired_vs_intrinsic_classification": classification,
        "classification_counts": dict(counts),
        "missing_crispr_summaries": missing,
    }

    if n_cas >= 2 and (n - n_cas) >= 2:
        mw_burden = mannwhitney_report(cas_pos["acquired_gene_count"], cas_neg["acquired_gene_count"],
                                        "acquired_gene_count: Cas-present vs Cas-absent")
        median_burden = merged["acquired_gene_count"].median()
        high_burden = merged["acquired_gene_count"] > median_burden
        fisher_highburden = fisher_report(
            int((merged["cas_system_present"] & high_burden).sum()),
            int((merged["cas_system_present"] & ~high_burden).sum()),
            int((~merged["cas_system_present"] & high_burden).sum()),
            int((~merged["cas_system_present"] & ~high_burden).sum()),
            f"cas_present vs above-median acquired burden (median={median_burden})",
        )
        fisher_resistance = fisher_report(
            int((merged["cas_system_present"] & (merged["is_resistant"] == 1)).sum()),
            int((merged["cas_system_present"] & (merged["is_resistant"] == 0)).sum()),
            int((~merged["cas_system_present"] & (merged["is_resistant"] == 1)).sum()),
            int((~merged["cas_system_present"] & (merged["is_resistant"] == 0)).sum()),
            f"cas_present vs {resistance_label_name} phenotype",
        )
        direction = ("HIGHER in Cas-present (contradicts literature)" if mw_burden.get("median_pos", 0) > mw_burden.get("median_neg", 0)
                     else "LOWER in Cas-present (supports literature)" if mw_burden.get("median_pos", 1) < mw_burden.get("median_neg", 1)
                     else "no difference")
        result.update({
            "mannwhitney_acquired_burden_cas_present_vs_absent": mw_burden,
            "direction_vs_literature": direction,
            "fisher_cas_present_vs_high_acquired_burden": fisher_highburden,
            f"fisher_cas_present_vs_{resistance_label_name}_phenotype": fisher_resistance,
        })
    else:
        note = (f"NOT COMPUTED -- only {n_cas} Cas-present / {n - n_cas} Cas-absent genome(s); "
                f"a 2x2 test with a near-zero cell is not statistically meaningful")
        result["statistical_battery"] = note
        print(f"  {note}")

    merged.drop(columns=["cas_subtypes"]).to_csv(OUT_DIR / f"{species_tag}_crispr_amr_merged.csv", index=False)
    (OUT_DIR / f"{species_tag}_crispr_amr_stats.json").write_text(json.dumps(result, indent=2, default=str))
    print(f"  Saved: {species_tag}_crispr_amr_merged.csv, {species_tag}_crispr_amr_stats.json")
    return result


if __name__ == "__main__":
    pa_result = run_paeruginosa()
    sa_result = run_saureus()
    ab_result = run_bvbrc_species("abaumannii", "A. baumannii",
                                   PROJECT_ROOT / "data/processed/abaumannii_features.csv",
                                   PROJECT_ROOT / "data/raw/bvbrc_abaumannii/crispr_sample_labels.json",
                                   "carbapenem")
    ef_result = run_bvbrc_species("efaecium", "E. faecium",
                                   PROJECT_ROOT / "data/processed/efaecium_features.csv",
                                   PROJECT_ROOT / "data/raw/bvbrc_efaecium/crispr_sample_labels.json",
                                   "vancomycin")
    ec_result = run_bvbrc_species("ecloacae", "E. cloacae",
                                   PROJECT_ROOT / "data/processed/ecloacae_features.csv",
                                   PROJECT_ROOT / "data/raw/bvbrc_ecloacae/crispr_sample_labels.json",
                                   "ESC-cephalosporin")
    print("\n=== DONE ===")

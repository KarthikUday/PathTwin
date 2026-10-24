"""
Curated, per-stage methods/limitations summaries for the dashboard.

These are hand-condensed from DECISIONS_AND_LIMITATIONS.md and README.md
-- NOT auto-extracted -- so each entry names the exact section of
DECISIONS_AND_LIMITATIONS.md it summarizes, letting a viewer go read the
full account rather than trusting a lossy condensation at face value.
"""

STAGE_METHODS = {
    1: {
        "summary": (
            "CARD's Resistance Gene Identifier (RGI) run against each genome's own assembly. "
            "'Cut_Off' is RGI's own confidence tier (Perfect/Strict/Loose); only Perfect and Strict "
            "hits are shown here (Loose hits are RGI's own low-confidence tier and are excluded from "
            "every count in this project). Intrinsic, species-defining genes (efflux pumps, the "
            "species' own core chromosomal beta-lactamase family) are real RGI hits, not evidence of "
            "acquired resistance by themselves -- see the acquired-vs-intrinsic distinction used "
            "throughout Stages 5-6."
        ),
        "limitations": [
            "Two ESKAPE species (E. faecium, E. cloacae) were initially picked by RefSeq's "
            "\"reference genome\" flag alone, which pointed to a clinical/surveillance isolate "
            "rather than the genuine type strain -- caught on review and corrected "
            "(see \"ESKAPE type-strain selection error\").",
            "3 of 7 diversity-strain accessions were wrong on first pull (ACICU/AYE swapped; "
            "E. faecium DO's accession was actually E. faecalis OG1RF; E. cloacae EcWSU1's "
            "accession was an unrelated Rhizobium) -- all independently verified against NCBI "
            "and corrected before use (see \"Diversity-strain accession errors\").",
            "Gene-family membership alone doesn't distinguish intrinsic from acquired "
            "(e.g. OXA-51-like is A. baumannii's intrinsic family; OXA-23/24/58-like are acquired "
            "carbapenemase families) -- see \"Recurring distinction: intrinsic gene-family "
            "membership vs. acquired resistance\".",
        ],
    },
    2: {
        "summary": (
            "AutoDock Vina docking of each target's real ligand against a real PDB structure, "
            "validated against that structure's own deposited co-crystal geometry (distance from "
            "the top docked pose to the catalytic residue, and to the real crystal ligand's own "
            "position) -- not a blind docking exercise with no ground truth to check against."
        ),
        "limitations": [
            "Several deposited ligands are not the intact administered drug: OXA-23's MER is "
            "meropenem's ring-opened hydrolysis PRODUCT; PDC-1's NXL and PBP5's PNM are covalent "
            "reaction adducts, mechanistically expected for these serine-hydrolase mechanisms, "
            "not a data quality problem -- intact drug was independently fetched from PubChem and "
            "docked non-covalently in every such case.",
            "PDC-1's originally-requested structure (4GZB) turned out fully apo (no ligand at "
            "all) -- substituted with 4HEF (genuinely avibactam-bound) after an RCSB search "
            "(see \"PDC-1/AmpC -- the originally-requested structure turned out apo\").",
            "P99/AmpC represents CMH-4's gene family via its best-characterized relative (P99), "
            "not CMH-4 itself -- CMH-4 (the allele Stage 1 actually detected) has no solved "
            "structure.",
            "SHV-1/tazobactam has no structured validation_summary.json (it was the first/"
            "baseline target, before this project adopted that file convention) -- shown here "
            "parsed from its raw Vina log; full validation detail is in README prose only.",
            "Vina cannot model covalent bond formation -- every covalent-mechanism target here "
            "(PDC-1, PBP5, OXA-23) is a standard non-covalent docking validation against the "
            "reacted/derivative crystal ligand, not a simulation of the true covalent step.",
        ],
    },
    3: {
        "summary": (
            "Real variant calling via snippy against each species' own reference genome, filtered "
            "against CARD's curated resistance-conferring SNPs by matching on (gene, AA position, "
            "wild-type residue, mutant residue) -- never on raw genomic coordinates, which differ "
            "between any two reference genomes."
        ),
        "limitations": [
            "CARD's SNP curation is manually curated, not exhaustive -- a real substitution at a "
            "well-known resistance hotspot can exist without being in CARD's own curated allele "
            "list for that exact position (see \"Honest curation-gap finding: gyrA S83Y\").",
            "Zero curated hits in several comparisons (both S. aureus pairs, E. faecium, "
            "E. cloacae) is a genuine, expected negative -- those species' real acquired genes "
            "(mecA, tet(M)/tet(L), DHA-1) are whole-gene insertions, not point mutations, so there "
            "is structurally nothing for this SNP-position-matching method to catch; RGI (Stage 1) "
            "already detected all of them by presence.",
        ],
    },
    4: {
        "summary": (
            "FoldX ΔΔG (structural stability) and AutoDock Vina re-docking (binding affinity), run "
            "for each mutation against both the wild-type and mutant structure. 'Both signals "
            "agree' (positive ΔΔG and positive Vina delta, i.e. weaker mutant binding) is the "
            "expected direction for a real resistance mutation; disagreement is reported as an "
            "informative negative, not smoothed over."
        ),
        "limitations": [
            "FoldX's BuildModel module is substitution-only -- it cannot model an insertion "
            "(PBP5's real literature mutation includes a serine insertion at position 466' that "
            "cannot be built at all; only the T485A substitution component was tested).",
            "PBP5 is the one mutation where the two signals disagree (ddG +0.131, Vina delta "
            "-0.702) -- read as an informative negative reflecting the untested partial genotype, "
            "not as evidence against the real literature finding.",
            "P99/AmpC's L293P was first tested against the wrong substrate (cephalothin, "
            "ambiguous result) and corrected to cefepime (the literature's actual substrate) -- "
            "only the corrected cefepime result has a surviving result.json; the original "
            "cephalothin numbers exist only in README/DECISIONS_AND_LIMITATIONS.md prose.",
            "OXA-23/OXA-239's real mutant crystal structures (5WIB/5WI3/5WI7) were rejected as "
            "confounded (extra K82D mutation on the same catalytic residue) -- FoldX modeling used "
            "instead of a real-but-confounded structure.",
        ],
    },
    5: {
        "summary": (
            "Random Forest + XGBoost trained on real genotype/phenotype data (KlebNET-GSP for "
            "K. pneumoniae; BV-BRC REST API sp_gene features for the other 5 species), with "
            "grouped train/test splits (by ST and by BioProject/study) used from the start for "
            "every species after K. pneumoniae, specifically to avoid clonal or study-batch "
            "leakage inflating accuracy."
        ),
        "limitations": [
            "K. pneumoniae's own grouped-split leakage check found no clonal leakage but a real, "
            "modest study/batch effect (~1-1.4 points) -- see the leakage-check panel below.",
            "Real, uncurated BV-BRC data repeatedly produced severe group imbalance (one dominant "
            "clone or source study) that degenerates a naive single-fold grouped split -- seen in "
            "P. aeruginosa (57% one BioProject), A. baumannii (46-58%), and E. faecium (43%). "
            "A. baumannii's original 1.000 accuracy was a real inflated artifact of this; a "
            "properly-designed holdout (dominant clone forced into train) gives ~0.80-0.87, the "
            "honest figure -- see the dominant-group follow-up panel.",
            "P. aeruginosa's fluoroquinolone extension found no clean net accuracy/AUC win from "
            "adding regulatory-SNP features, despite real, strengthening SHAP signal (oprD_lof "
            "rank 11->4 at scale) -- reported as a genuinely mixed, algorithm-dependent result, "
            "not resolved toward a cleaner story.",
            "E. cloacae's classifier shows zero expected AmpC markers in its SHAP top-15 -- a real "
            "negative, verified as mechanistically expected: the AmpC gene family is ~100% "
            "prevalent regardless of phenotype (resistance here is regulatory derepression, "
            "invisible to a presence/absence feature).",
        ],
    },
    6: {
        "summary": (
            "CRISPRCasTyper (cctyper) run on each genome assembly; a CRISPR-Cas system is only "
            "counted 'present' when a Cas operon clears cctyper's own confidence threshold "
            "(cas_operons.tab has rows) -- a weak/candidate hit alone does not count. "
            "Acquired-resistance burden = count of gene-presence features classified 'acquired' "
            "(not intrinsic/core) per genome, cross-checked against real RGI output for every "
            "borderline family call."
        ),
        "limitations": [
            "The original n=3-15 proof-of-concept scale could not support any real statistical "
            "test -- results at that scale are qualitative reads, not p-values.",
            "P. aeruginosa's original 2-genome (PAO1 vs PA14) 'AYE-style contradiction' reversed "
            "into strong literature SUPPORT once scaled to 1020 genomes (Mann-Whitney p=2.9e-12).",
            "A. baumannii's population-wide result at n=382 looked like a clean null, but one "
            "dominant clone (44% of the sample, 100% uniform on CRISPR-Cas status) was diluting a "
            "real signal via clonal pseudo-replication -- excluding it reveals a strong, "
            "significant CONTRADICTION of the literature (p=7.1e-8) -- see the dominant-clone "
            "panel for that species.",
            "S. aureus (0.75%) and E. faecium (1.2%) are both genuinely CRISPR-Cas-poor at real "
            "population scale, not just undersampled -- the correlation is untestable there for a "
            "confirmed biological reason, not a data gap.",
            "E. cloacae's n=106 is real but its label distribution (97.3% Resistant) degenerates "
            "the phenotype-correlation test regardless of CRISPR-Cas status -- flagged as the "
            "least precise of the species tested, not presented at the same confidence as the "
            "others.",
            "CRISPR-Cas subtype does not predict the direction of the burden correlation across "
            "species -- the same type I-F subtype associates with LOWER burden in P. aeruginosa "
            "and HIGHER burden in A. baumannii once deconfounded.",
        ],
    },
}

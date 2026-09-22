"""
PathTwin Results Dashboard -- data loading layer.

Every function here reads directly from an existing file under results/
(or data/processed/ for a couple of Stage 5 metadata files) and returns
either the parsed data or None plus the path it looked for -- the app
layer uses that to render an explicit "not found" state rather than
silently omitting a section. Nothing in this module recomputes,
re-derives, or interpolates a number; it only parses what's already on
disk.
"""
import csv
import json
import re
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

SPECIES = {
    "K. pneumoniae": "kp",
    "S. aureus": "sa",
    "A. baumannii": "ab",
    "P. aeruginosa": "pa",
    "E. faecium": "ef",
    "E. cloacae": "ec",
}


def _exists(path: Path):
    return path if path.exists() else None


# ----------------------------- Stage 1: RGI -----------------------------

RGI_COLUMNS = [
    "ORF_ID", "Contig", "Cut_Off", "Best_Hit_ARO", "Best_Identities",
    "Drug Class", "Resistance Mechanism", "AMR Gene Family",
]

# (display_label, path) for every genome Stage 1 actually ran RGI against.
# Hand-enumerated from the real files on disk, not globbed blindly, so a
# renamed/missing file surfaces as a clear gap rather than silently
# vanishing from a glob.
STAGE1_GENOMES = [
    ("K. pneumoniae ATCC 13883 (type strain)", RESULTS_DIR / "kp_atcc13883_rgi.txt"),
    ("S. aureus NCTC 8325 (type/reference strain)", RESULTS_DIR / "sa_nctc8325_rgi.txt"),
    ("S. aureus USA300-FPR3757 (diversity)", RESULTS_DIR / "rgi_diversity" / "saureus_usa300_rgi.txt"),
    ("S. aureus MRSA252 (diversity)", RESULTS_DIR / "rgi_diversity" / "saureus_mrsa252_rgi.txt"),
    ("A. baumannii ATCC 19606 (type strain)", RESULTS_DIR / "ab_atcc19606_rgi.txt"),
    ("A. baumannii ACICU (diversity)", RESULTS_DIR / "rgi_diversity" / "abaumannii_acicu_rgi.txt"),
    ("A. baumannii AYE (diversity)", RESULTS_DIR / "rgi_diversity" / "abaumannii_aye_rgi.txt"),
    ("P. aeruginosa PAO1 (reference strain)", RESULTS_DIR / "pa_pao1_rgi.txt"),
    ("P. aeruginosa PA14 (diversity)", RESULTS_DIR / "rgi_diversity" / "paeruginosa_pa14_rgi.txt"),
    ("E. faecium DSM 20477 (type strain)", RESULTS_DIR / "ef_dsm20477_rgi.txt"),
    ("E. faecium DO (diversity)", RESULTS_DIR / "rgi_diversity" / "efaecium_do_rgi.txt"),
    ("E. cloacae ATCC 13047 (type strain)", RESULTS_DIR / "ec_atcc13047_rgi.txt"),
    ("E. cloacae 9553 (diversity)", RESULTS_DIR / "rgi_diversity" / "ecloacae_9553_rgi.txt"),
]


def stage1_genomes_for_species(species: str):
    return [(label, path) for label, path in STAGE1_GENOMES if label.startswith(species) or species in label]


CUT_OFF_ORDER = ["Perfect", "Strict", "Loose"]


def load_rgi_table(path: Path):
    """Returns a DataFrame of the readable RGI columns, or None if the
    file doesn't exist. Empty file (0 hits) returns an empty DataFrame,
    not None -- that's a real, meaningful result, not a missing one.

    Default-sorted Perfect -> Strict -> Loose, then by Best_Identities
    descending within each tier -- the values themselves are untouched
    (Best_Identities parsed to float only for sorting; the displayed
    column keeps its original string), this only orders real rows, it
    doesn't recompute or filter anything."""
    if not path.exists():
        return None
    df = pd.read_csv(path, sep="\t", dtype=str)
    cols = [c for c in RGI_COLUMNS if c in df.columns]
    df = df[cols].reset_index(drop=True)
    if df.empty:
        return df
    if "Cut_Off" in df.columns:
        df["_cutoff_rank"] = df["Cut_Off"].apply(
            lambda v: CUT_OFF_ORDER.index(v) if v in CUT_OFF_ORDER else len(CUT_OFF_ORDER))
    else:
        df["_cutoff_rank"] = 0
    df["_identity_sort"] = pd.to_numeric(df.get("Best_Identities"), errors="coerce")
    df = df.sort_values(["_cutoff_rank", "_identity_sort"], ascending=[True, False])
    return df.drop(columns=["_cutoff_rank", "_identity_sort"]).reset_index(drop=True)


# --------------------------- Stage 2: docking ---------------------------

DOCKING_TARGETS = [
    ("SHV-1 / tazobactam", "shv1_tazobactam", "K. pneumoniae"),
    ("PBP3 (FtsI) / JXJ", "pbp3_jxj", "K. pneumoniae"),
    ("FosA / fosfomycin", "fosa_fosfomycin", "K. pneumoniae"),
    ("PBP2a / cefepime", "pbp2a_cefepime", "S. aureus"),
    ("OXA-23 / meropenem", "oxa23_meropenem", "A. baumannii"),
    ("PDC-1/AmpC / avibactam", "pdc1_avibactam", "P. aeruginosa"),
    ("PBP5 / benzylpenicillin", "pbp5_benzylpenicillin", "E. faecium"),
    ("P99/AmpC / cephalothin", "p99_cephalothin", "E. cloacae"),
]

# Wet-lab candidate validations -- a different kind of test from the 8
# pipeline-validation targets above (does a real bioactive compound dock
# favorably at an already-validated site, not "does the pipeline
# reproduce a target's own known ligand"). Kept as its own list/section
# rather than folded into DOCKING_TARGETS.
WETLAB_CANDIDATES = [
    ("PPDHMP (cyclo(L-Leu-L-Pro)) @ PBP2a allosteric site", "pbp2a_ppdhmp",
     "S. aureus", "PBP2a / cefepime"),  # (label, dir, species, compared_against)
    ("Chloramphenicol @ PBP2a allosteric site (likely reference compound)", "pbp2a_chloramphenicol",
     "S. aureus", "PBP2a / cefepime"),
    ("(2E,5E)-Phenyltetradeca-2,5-dienoate @ PDC-1/AmpC active site", "pdc1_phenyltetradecadienoate",
     "P. aeruginosa", "PDC-1 / avibactam"),
]


# ------------------- Docking-vs-MIC correlation test (exploratory) -------------------

def load_mic_correlation_test():
    """Returns (raw_dict, source_path) for the cross-compound docking-vs-MIC
    correlation test (PPDHMP + 3 literature compounds), or (None, None) if
    it hasn't been run. Full raw JSON returned as-is -- this is a one-off,
    narrative-heavy result (comparison table, per-target Spearman attempts,
    an honest_summary explaining why no real statistic could be computed),
    not a schema shared with any other loader."""
    path = RESULTS_DIR / "docking" / "mic_correlation_test.json"
    if not path.exists():
        return None, None
    return json.loads(path.read_text()), str(path.relative_to(PROJECT_ROOT))


def load_docking_summary(target_dir: str):
    """Returns (record_dict, source_note) for one docking target.
    record_dict is always populated with whatever's available; source_note
    says exactly which file(s) it came from, since one target (SHV-1) has
    no structured validation_summary.json and is parsed from the raw Vina
    log instead."""
    d = RESULTS_DIR / "docking" / target_dir
    json_path = d / "validation_summary.json"
    if json_path.exists():
        raw = json.loads(json_path.read_text())
        dock = raw.get("docking", {})
        val = raw.get("validation", {})
        dist_keys = [k for k in val if k.startswith("top_pose_distance_to_")]
        closest_keys = [k for k in val if "closest_atom_to_crystal" in k]
        centroid_keys = [k for k in val if "centroid_to_centroid" in k]
        record = {
            "target": raw.get("target"),
            "pdb_id": raw.get("pdb_id"),
            "ligand": raw.get("ligand"),
            "nucleophile": raw.get("catalytic_site", {}).get("nucleophile"),
            "top_pose_affinity_kcal_mol": dock.get("top_pose_affinity_kcal_mol"),
            "primary_catalytic_distance_label": dist_keys[0] if dist_keys else None,
            "primary_catalytic_distance_A": val.get(dist_keys[0]) if dist_keys else None,
            "closest_atom_to_crystal_ligand_A": val.get(closest_keys[0]) if closest_keys else None,
            "centroid_to_centroid_vs_crystal_A": val.get(centroid_keys[0]) if centroid_keys else None,
            "conclusion": val.get("conclusion"),
        }
        return record, f"results/docking/{target_dir}/validation_summary.json"

    # Fallback: SHV-1 has no JSON summary -- parse the raw Vina log directly,
    # nothing invented for the fields that file doesn't contain.
    log_path = d / "vina_log.txt"
    if log_path.exists():
        text = log_path.read_text()
        m = re.search(r"^\s*1\s+(-?\d+\.\d+)", text, re.MULTILINE)
        affinity = float(m.group(1)) if m else None
        record = {
            "target": "SHV-1 beta-lactamase / tazobactam",
            "pdb_id": "1SHV",
            "ligand": "tazobactam",
            "nucleophile": None,
            "top_pose_affinity_kcal_mol": affinity,
            "primary_catalytic_distance_label": None,
            "primary_catalytic_distance_A": None,
            "closest_atom_to_crystal_ligand_A": None,
            "centroid_to_centroid_vs_crystal_A": None,
            "conclusion": ("No structured validation_summary.json exists for this target -- it was "
                           "the first/baseline Stage 2 target, run before this project adopted the "
                           "JSON summary convention (see README's Stage 2 section). Top-pose affinity "
                           "parsed directly from the raw Vina log; distance-to-catalytic-residue and "
                           "centroid metrics are documented in prose in README, not machine-readable here."),
        }
        return record, f"results/docking/{target_dir}/vina_log.txt (parsed; no JSON summary exists)"

    return None, None


def load_accuracy_benchmark():
    """Returns (per_target_dict, source_path) for the active-vs-decoy
    docking accuracy benchmark (AUC-ROC / EF1%), or (None, None) if it
    hasn't been run. per_target_dict is keyed by target_dir, each value
    the target's own accuracy_summary.json content."""
    path = RESULTS_DIR / "docking" / "accuracy_benchmark" / "overall_summary.json"
    if not path.exists():
        return None, None
    return json.loads(path.read_text()), "results/docking/accuracy_benchmark/overall_summary.json"


def load_bootstrap_ci():
    """Returns (dict, source_path) for the bootstrap 95% CI / pairwise
    significance analysis over the accuracy benchmark's AUCs, or
    (None, None) if it hasn't been run."""
    path = RESULTS_DIR / "docking" / "accuracy_benchmark" / "bootstrap_ci_summary.json"
    if not path.exists():
        return None, None
    return json.loads(path.read_text()), "results/docking/accuracy_benchmark/bootstrap_ci_summary.json"


def load_adduct_vs_free_comparison():
    """Returns (dict, source_path) for the covalent-adduct-vs-free-drug
    AUC comparison (4 targets re-run with their real ring-opened/
    covalently-engaged form), or (None, None) if it hasn't been run."""
    path = RESULTS_DIR / "docking" / "accuracy_benchmark" / "adduct_vs_free_comparison.json"
    if not path.exists():
        return None, None
    return json.loads(path.read_text()), "results/docking/accuracy_benchmark/adduct_vs_free_comparison.json"


def load_sfct_investigation():
    """Returns (dict, source_path) for the SwissDock-feasibility /
    OnionNet-SFCT-rescoring investigation at SHV-1, or (None, None) if
    it hasn't been run."""
    path = RESULTS_DIR / "docking" / "sfct_rescoring" / "shv1_covalent_docking_investigation.json"
    if not path.exists():
        return None, None
    return json.loads(path.read_text()), "results/docking/sfct_rescoring/shv1_covalent_docking_investigation.json"


def load_shv1_box_correction():
    """Returns (dict, source_path) for the SHV-1 box-correction re-run
    + 8-target apo-receptor audit, or (None, None) if not yet run."""
    path = RESULTS_DIR / "docking" / "accuracy_benchmark" / "shv1_box_correction_and_audit.json"
    if not path.exists():
        return None, None
    return json.loads(path.read_text()), "results/docking/accuracy_benchmark/shv1_box_correction_and_audit.json"


def load_wetlab_candidate(target_dir: str):
    """Returns (full_raw_dict, source_path) for a wet-lab candidate
    validation -- unlike load_docking_summary above, this returns the
    full raw JSON rather than a flattened record, since these have a
    richer, one-off schema (identity verification, ligand-efficiency
    comparison, literature citations) worth showing in full."""
    path = RESULTS_DIR / "docking" / target_dir / "validation_summary.json"
    if not path.exists():
        return None, None
    return json.loads(path.read_text()), str(path.relative_to(PROJECT_ROOT))


# --------------------------- Stage 3: mutations ---------------------------

def list_mutation_comparisons():
    """(label, dir_path) for every real Stage 3 snippy comparison on disk."""
    dirs = sorted(RESULTS_DIR.glob("*_mutations"))
    return [(d.name.replace("_", " "), d) for d in dirs]


def load_mutation_comparison(dir_path: Path):
    """Returns dict with raw_variant_count, curated_hits (DataFrame), and
    source paths. None fields if a file is missing."""
    snps_tab = dir_path / "snps.tab"
    curated_tab = dir_path / "resistance_relevant_snps.tab"
    raw_count = None
    if snps_tab.exists():
        with open(snps_tab, newline="") as f:
            raw_count = sum(1 for _ in f) - 1  # header row
    curated_df = None
    if curated_tab.exists():
        curated_df = pd.read_csv(curated_tab, sep="\t", dtype=str)
    return {
        "raw_variant_count": raw_count,
        "raw_source": str(snps_tab.relative_to(PROJECT_ROOT)) if snps_tab.exists() else None,
        "curated_hits": curated_df,
        "curated_source": str(curated_tab.relative_to(PROJECT_ROOT)) if curated_tab.exists() else None,
    }


# --------------------------- Stage 4: mutation_sim ---------------------------

# (mutation_dir, display_label, target, species, caveat) -- caveat text is
# hand-curated from DECISIONS_AND_LIMITATIONS.md / README for the handful
# of mutations with a real, stated modeling limitation; left blank
# otherwise. Every ddG/Vina number itself still comes only from the real
# result.json below -- this only adds the documented context for reading it.
STAGE4_MUTATIONS = [
    ("A_238_G_S", "SHV-1 G238S", "SHV-1 beta-lactamase", "K. pneumoniae", ""),
    ("A_179_D_N", "SHV-1 D179N", "SHV-1 beta-lactamase", "K. pneumoniae", ""),
    ("A_367_L_Q", "PBP3 L367Q", "PBP3 (FtsI)", "K. pneumoniae", ""),
    ("A_247_E_K", "PDC-1/AmpC E247K", "PDC-1/AmpC", "P. aeruginosa",
     "Literature's \"E219K\" uses a standardized class-C numbering scheme that doesn't match "
     "4HEF's own residue 219 (Pro, not Glu) -- resolved by locating the paper's own flanking "
     "sequence motif directly in 4HEF's real sequence (found at residue 247)."),
    ("A_166_E_A", "PBP2a N146K/E150K (docked, no FoldX)", "PBP2a", "S. aureus",
     "Two real crystal structures (5M18 vs. the clinical double mutant 4CPK) were docked directly "
     "instead of using FoldX -- this is the strongest evidence class in the project (two real "
     "structures compared directly, not a modeled mutant)."),
    ("SA109L_DA222N_PA225S", "OXA-23/OXA-239 S109L+D222N+P225S", "OXA-23", "A. baumannii",
     "Real OXA-239 crystal structures exist (5WIB/5WI3/5WI7) but all carry an extra K82D mutation "
     "on the same catalytic Lys82 central to this target, so they were rejected as confounded; "
     "FoldX's --multi-mutation flag applied all 3 substitutions as one combined mutant instead."),
    ("A_485_T_A", "PBP5 T485A (proxy for M485A)", "PBP5", "E. faecium",
     "6MKG carries a natural Thr485 polymorphism vs. the literature reference's Met485 (verified "
     "by direct sequence alignment) -- T485A was used as the closest available proxy. The "
     "literature's serine-insertion component (position 466') cannot be modeled at all: FoldX's "
     "BuildModel module is substitution-only, a genuine tool-capability limit, not a shortcut "
     "taken here. Read this result as a partial-genotype test, not a full reproduction of the "
     "literature mutation."),
    ("A_293_L_P", "P99/AmpC L293P (cefepime, corrected)", "P99/AmpC", "E. cloacae",
     "This is the corrected re-run. The first run used cephalothin (this target's own validated "
     "Stage 2 ligand) and came back ambiguous (ddG +0.074, Vina delta -0.265, signals disagreeing) "
     "-- not wrong, just the wrong substrate: L293P's real literature mechanism (Barnaud et al. "
     "2004) is specifically an improved kcat/Km for cefepime. That first cephalothin run's numbers "
     "are documented in README/DECISIONS_AND_LIMITATIONS.md prose only -- no result.json for it "
     "was kept on disk (overwritten by this corrected run), so only this cefepime result is shown "
     "here as a real, file-backed number."),
]


def load_mutation_sim_result(mutation_dir: str):
    path = RESULTS_DIR / "mutation_sim" / mutation_dir / "result.json"
    if not path.exists():
        return None, None
    return json.loads(path.read_text()), str(path.relative_to(PROJECT_ROOT))


# --------------------------- Stage 5: ML classifiers ---------------------------

STAGE5_CLASSIFIERS = [
    ("K. pneumoniae", "ciprofloxacin", RESULTS_DIR / "ml_classifier" / "stage5_results.json",
     RESULTS_DIR / "ml_classifier" / "shap_summary.png"),
    ("S. aureus", "oxacillin", RESULTS_DIR / "ml_classifier" / "saureus_stage5_results.json",
     RESULTS_DIR / "ml_classifier" / "saureus_shap_summary.png"),
    ("P. aeruginosa", "fluoroquinolone", RESULTS_DIR / "ml_classifier" / "paeruginosa_fq_stage5_results.json",
     RESULTS_DIR / "ml_classifier" / "paeruginosa_combined_shap_summary.png"),
    ("A. baumannii", "carbapenem", RESULTS_DIR / "ml_classifier" / "abaumannii_carbapenem_stage5_results.json",
     RESULTS_DIR / "ml_classifier" / "abaumannii_carbapenem_shap_summary.png"),
    ("E. faecium", "vancomycin", RESULTS_DIR / "ml_classifier" / "efaecium_vancomycin_stage5_results.json",
     RESULTS_DIR / "ml_classifier" / "efaecium_vancomycin_shap_summary.png"),
    ("E. cloacae", "ESC-cephalosporin", RESULTS_DIR / "ml_classifier" / "ecloacae_esc_stage5_results.json",
     RESULTS_DIR / "ml_classifier" / "ecloacae_esc_shap_summary.png"),
]


def load_stage5_result(json_path: Path):
    if not json_path.exists():
        return None
    return json.loads(json_path.read_text())


def load_stage5_leakage_check():
    path = RESULTS_DIR / "ml_classifier" / "stage5_grouped_split_comparison.json"
    if not path.exists():
        return None, None
    return json.loads(path.read_text()), str(path.relative_to(PROJECT_ROOT))


def load_dominant_group_investigation():
    path = RESULTS_DIR / "ml_classifier" / "dominant_group_investigation.json"
    if not path.exists():
        return None, None
    return json.loads(path.read_text()), str(path.relative_to(PROJECT_ROOT))


# --------------------------- Stage 6: CRISPR-Cas ---------------------------

STAGE6_SPECIES = [
    ("K. pneumoniae", None),  # only in the original n=15 crossref, no dedicated stats JSON
    ("S. aureus", RESULTS_DIR / "crispr_scan" / "saureus_crispr_amr_stats.json"),
    ("P. aeruginosa", RESULTS_DIR / "crispr_scan" / "paeruginosa_crispr_amr_stats.json"),
    ("A. baumannii", RESULTS_DIR / "crispr_scan" / "abaumannii_crispr_amr_stats.json"),
    ("E. faecium", RESULTS_DIR / "crispr_scan" / "efaecium_crispr_amr_stats.json"),
    ("E. cloacae", RESULTS_DIR / "crispr_scan" / "ecloacae_crispr_amr_stats.json"),
]


def load_stage6_result(path):
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text())


def load_stage6_original_crossref():
    path = RESULTS_DIR / "crispr_scan" / "crispr_amr_crossref_full_eskape.json"
    if not path.exists():
        return None, None
    return json.loads(path.read_text()), str(path.relative_to(PROJECT_ROOT))


def load_stage6_merged_csv(species_tag: str):
    path = RESULTS_DIR / "crispr_scan" / f"{species_tag}_crispr_amr_merged.csv"
    if not path.exists():
        return None, None
    return pd.read_csv(path), str(path.relative_to(PROJECT_ROOT))


# --------------------------- Coverage matrix (Overview) ---------------------------

_MUTATION_DIR_SPECIES_PREFIX = {
    "kp": "K. pneumoniae", "sa": "S. aureus", "ab": "A. baumannii",
    "pa": "P. aeruginosa", "ef": "E. faecium", "ec": "E. cloacae",
}


def compute_coverage_matrix():
    """Real file-existence-based coverage per (species, stage) -- every
    cell traces to an actual check against disk, not an assumption from
    a hardcoded list alone. Returns a DataFrame: rows=species,
    columns=Stage 1..6, values True/False/"partial"."""
    species_list = list(SPECIES.keys())
    matrix = {sp: {i: False for i in range(1, 7)} for sp in species_list}

    # Stage 1
    for sp in species_list:
        genomes = stage1_genomes_for_species(sp)
        matrix[sp][1] = any(path.exists() for _, path in genomes)

    # Stage 2 (target -> species)
    for _, target_dir, sp in DOCKING_TARGETS:
        rec, _ = load_docking_summary(target_dir)
        if rec is not None:
            matrix[sp][2] = True

    # Stage 3 (dir name prefix -> species)
    for label, dir_path in list_mutation_comparisons():
        prefix = dir_path.name.split("_")[0]
        sp = _MUTATION_DIR_SPECIES_PREFIX.get(prefix)
        if sp and (dir_path / "snps.tab").exists():
            matrix[sp][3] = True

    # Stage 4
    for mdir, _, _, sp, _ in STAGE4_MUTATIONS:
        result, _ = load_mutation_sim_result(mdir)
        if result is not None:
            matrix[sp][4] = True

    # Stage 5
    for sp, _, json_path, _ in STAGE5_CLASSIFIERS:
        if load_stage5_result(json_path) is not None:
            matrix[sp][5] = True

    # Stage 6
    crossref, _ = load_stage6_original_crossref()
    kp_rows = [g for g in crossref["genomes"] if g["species"] == "K. pneumoniae"] if crossref else []
    for sp, path in STAGE6_SPECIES:
        if sp == "K. pneumoniae":
            matrix[sp][6] = "partial" if kp_rows else False
        elif load_stage6_result(path) is not None:
            matrix[sp][6] = True

    rows = []
    for sp in species_list:
        row = {"Species": sp}
        row.update({f"Stage {i}": matrix[sp][i] for i in range(1, 7)})
        rows.append(row)
    return pd.DataFrame(rows)

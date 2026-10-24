#!/usr/bin/env python3
"""
PathTwin Stage 2 accuracy validation -- decoy generation for the covalent-
ADDUCT re-run: same method as generate_all_decoys.py (DUD-E-inspired
property-matched + topologically-dissimilar PubChem decoys), but matched
to each target's real ring-opened/covalently-engaged adduct form (the
molecule actually observed bound in the crystal structure) instead of
the free pre-reaction drug.

Only 4 targets get an adduct re-run -- decided by direct investigation,
not assumption (see DECISIONS_AND_LIMITATIONS.md "Covalent-adduct
active-vs-decoy re-run" for the full account of what was checked and why
the other 4 targets were excluded):

  - SHV-1: TBE ("TAZOBACTAM INTERMEDIATE", PDB entry 1VM1 -- SHV-1's own
    real tazobactam-inhibited co-crystal structure; 1SHV itself is apo).
    Formula C10H14N4O5S vs. intact tazobactam C10H12N4O5S: +2H, the same
    ring-opening signature as every other adduct in this project.
  - OXA-23: MER, the already-identified ring-opened meropenem hydrolysis
    product (C17H27N3O5S vs. meropenem's C17H25N3O5S: +2H).
  - PDC-1: NXL, the already-identified ring-opened avibactam covalent
    form (C7H13N3O6S vs. avibactam's C7H11N3O6S: +2H).
  - PBP5: PNM, the already-identified ring-opened benzylpenicillin acyl-
    adduct (C16H20N2O4S vs. benzylpenicillin's C16H18N2O4S: +2H).

Excluded, confirmed by direct investigation, not left to assumption:
  - PBP3 (JXJ): checked RCSB's own chemcomp record for JXJ -- formula
    C14H19N5O9S2, IDENTICAL to what this project already fetched and
    docked as JXJ's "active" in the original benchmark. There is no
    separate free/adduct distinction for this target; JXJ already *is*
    the covalently-observed form. No re-run needed or possible.
  - P99 (cephalothin): checked RCSB's own chemcomp record for IPP (the
    deposited 1BLS ligand) -- formula C9H11INO3P, containing iodine and
    phosphorus with no chemical relationship to cephalothin (C16H16N2O6S2).
    IPP is a designed phosphonate transition-state mimic, not a reacted
    form of cephalothin -- no real "cephalothin adduct" structure exists
    to test.
  - FosA, PBP2a: excluded per the original request -- both established
    non-covalent mechanisms.

Usage: python3 scripts/generate_adduct_decoys.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_decoys import generate_decoys, embed_3d  # noqa: E402
from rdkit import Chem  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = PROJECT_ROOT / "results" / "docking" / "accuracy_benchmark"

# (adduct_target_dir, adduct_name, adduct_smiles, free_target_dir_for_reference)
ADDUCT_TARGETS = [
    ("shv1_tazobactam_adduct", "TBE",
     r"C[C@](Cn1ccnn1)([C@@H](N/C=C\C=O)C(=O)O)[SH](=O)=O", "shv1_tazobactam"),
    ("oxa23_meropenem_adduct", "MER",
     "C[C@@H](O)[C@@H](C=O)[C@@H]1NC(C(=O)O)=C(S[C@@H]2CN[C@H](C(=O)N(C)C)C2)[C@@H]1C",
     "oxa23_meropenem"),
    ("pdc1_avibactam_adduct", "NXL",
     "NC(=O)[C@@H]1CC[C@@H](NOS(=O)(=O)O)CN1C=O", "pdc1_avibactam"),
    ("pbp5_benzylpenicillin_adduct", "PNM",
     "CC1(C)S[C@H]([C@@H](C=O)NC(=O)Cc2ccccc2)N[C@H]1C(=O)O", "pbp5_benzylpenicillin"),
]

N_DECOYS = 30
SEED = 42


def main():
    for target_dir, adduct_name, smiles, free_ref in ADDUCT_TARGETS:
        print(f"\n############ {target_dir} ({adduct_name}, adduct form of {free_ref}) ############",
              flush=True)
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"  [ERROR] could not parse adduct SMILES for {target_dir}: {smiles}")
            continue
        out_dir = OUT_ROOT / target_dir
        (out_dir / "decoy_sdf").mkdir(parents=True, exist_ok=True)
        manifest_path = out_dir / "decoy_manifest.json"

        import json

        # Resume support, same fix pattern as crispr_scan.py's 2026-09-04
        # cache-completeness bug: decoy_manifest.json is written BEFORE
        # the embedding loop below (the network/selection work finishes
        # first), so its mere existence is an early marker, not proof the
        # target's decoy generation actually finished -- a power cut
        # mid-embedding would leave a "complete-looking" manifest with
        # some SDFs missing. The real completion signal is the
        # `generation_complete` flag, written back into the manifest only
        # AFTER every decoy has been attempted for embedding.
        if manifest_path.exists():
            cached = json.loads(manifest_path.read_text())
            if cached.get("generation_complete"):
                n_sdf = len(list((out_dir / "decoy_sdf").glob("*.sdf")))
                print(f"  already complete ({cached.get('n_embedded', '?')} embedded, "
                      f"{n_sdf} SDF files on disk) -- skipping, not re-fetching from PubChem", flush=True)
                continue
            print("  found an incomplete prior manifest (no generation_complete flag) -- "
                  "re-running this target's decoy generation from scratch rather than trusting it",
                  flush=True)

        result = generate_decoys(smiles, N_DECOYS, seed=SEED)
        result["active_name"] = adduct_name
        result["is_adduct_form"] = True
        result["free_form_target_for_comparison"] = free_ref
        print(f"  found {result['n_found']} / {result['n_requested']}", flush=True)

        # Written now (pre-embedding) so a crash during embedding leaves
        # useful diagnostic content on disk, but generation_complete is
        # deliberately absent until the embedding loop below finishes --
        # that absence is exactly what the resume check above relies on.
        with open(manifest_path, "w") as f:
            json.dump(result, f, indent=2)

        n_embedded = 0
        for i, d in enumerate(result["decoys"]):
            dmol = Chem.MolFromSmiles(d["smiles"])
            mol3d = embed_3d(dmol, seed=SEED)
            if mol3d is None:
                print(f"  [warn] 3D embedding failed for CID {d['cid']}")
                continue
            sdf_path = out_dir / "decoy_sdf" / f"decoy_{i:02d}_cid{d['cid']}.sdf"
            writer = Chem.SDWriter(str(sdf_path))
            mol3d.SetProp("_Name", f"decoy_{i:02d}_cid{d['cid']}")
            writer.write(mol3d)
            writer.close()
            n_embedded += 1
        print(f"  3D-embedded {n_embedded} / {len(result['decoys'])} decoys", flush=True)

        # The true completion write: only reached after every decoy has
        # been attempted. This is what makes a resumed run trustworthy.
        result["generation_complete"] = True
        result["n_embedded"] = n_embedded
        with open(manifest_path, "w") as f:
            json.dump(result, f, indent=2)

    print("\nALL_ADDUCT_DECOY_GEN_DONE", flush=True)


if __name__ == "__main__":
    main()

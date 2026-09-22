#!/usr/bin/env python3
"""
Driver for generate_decoys.py across all 8 Stage 2 validated docking
targets -- kept as a separate driver (rather than a shell loop) because
the per-target active SMILES include backslash-escaped E/Z bond notation
(JXJ, cefepime) that is fragile to carry through nested shell quoting.

Run from the project root: python3 scripts/generate_all_decoys.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_decoys import generate_decoys, embed_3d  # noqa: E402
from rdkit import Chem  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = PROJECT_ROOT / "results" / "docking" / "accuracy_benchmark"

# (target_dir_name, active_name, active_smiles) -- SMILES taken directly
# from RDKit's own canonicalization of each target's real, already-docked
# active ligand (read back from its prep .sdf, or re-fetched from the same
# validated PubChem CID used originally for the 3 targets with no local
# .sdf) -- see decoy generation methodology note in DECISIONS_AND_LIMITATIONS.md.
TARGETS = [
    ("shv1_tazobactam", "tazobactam",
     "C[C@]1(Cn2ccnn2)[C@H](C(=O)O)N2C(=O)C[C@H]2S1(=O)=O"),
    ("pbp3_jxj", "JXJ",
     r"CC(C)(NOS(=O)(=O)O)[C@@H](C=O)NC(=O)/C(=N\OC1(C(=O)O)CC1)c1csc(N)n1"),
    ("fosa_fosfomycin", "fosfomycin",
     "C[C@@H]1O[C@@H]1P(=O)(O)O"),
    ("pbp2a_cefepime", "cefepime",
     r"CO/N=C(\C(=O)N[C@@H]1C(=O)N2C(C(=O)[O-])=C(C[N+]3(C)CCCC3)CS[C@H]12)c1csc(N)n1"),
    ("oxa23_meropenem", "meropenem",
     "C[C@@H](O)[C@H]1C(=O)N2C(C(=O)O)=C(S[C@@H]3CN[C@H](C(=O)N(C)C)C3)[C@H](C)[C@H]12"),
    ("pdc1_avibactam", "avibactam",
     "C1C[C@H](N2C[C@@H]1N(C2=O)OS(=O)(=O)O)C(=O)N"),
    ("pbp5_benzylpenicillin", "benzylpenicillin",
     "CC1([C@@H](N2[C@H](S1)[C@@H](C2=O)NC(=O)CC3=CC=CC=C3)C(=O)O)C"),
    ("p99_cephalothin", "cephalothin",
     "CC(=O)OCC1=C(N2[C@@H]([C@@H](C2=O)NC(=O)CC3=CC=CS3)SC1)C(=O)O"),
]

N_DECOYS = 30
SEED = 42


def main():
    for target_dir, active_name, smiles in TARGETS:
        print(f"\n############ {target_dir} ({active_name}) ############", flush=True)
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"  [ERROR] could not parse active SMILES for {target_dir}: {smiles}")
            continue
        out_dir = OUT_ROOT / target_dir
        (out_dir / "decoy_sdf").mkdir(parents=True, exist_ok=True)
        manifest_path = out_dir / "decoy_manifest.json"

        import json

        # Resume support, same fix pattern as crispr_scan.py's 2026-09-04
        # cache-completeness bug: decoy_manifest.json is written BEFORE
        # the embedding loop below, so its mere existence is an early
        # marker, not proof this target's decoy generation finished -- a
        # power cut mid-embedding would leave a "complete-looking"
        # manifest with some SDFs missing. The real completion signal is
        # `generation_complete`, written back only after every decoy has
        # been attempted for embedding.
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
        result["active_name"] = active_name
        print(f"  found {result['n_found']} / {result['n_requested']}", flush=True)

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

        result["generation_complete"] = True
        result["n_embedded"] = n_embedded
        with open(manifest_path, "w") as f:
            json.dump(result, f, indent=2)

    print("\nALL_DECOY_GEN_DONE", flush=True)


if __name__ == "__main__":
    main()

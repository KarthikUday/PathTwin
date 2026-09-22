#!/usr/bin/env python3
"""
PathTwin Stage 2 accuracy validation -- property-matched decoy generation,
inspired by the DUD-E (Directory of Useful Decoys, Enhanced) protocol
(Mysinger, Carchia, Irwin, Shoichet, J. Med. Chem. 2012, 55(14):6582-94,
PMID 22716043): for each target's own real active ligand, pull candidate
decoys from PubChem within a physicochemical window of the active
(molecular weight, logP) and keep only those that are topologically
DISSIMILAR (low 2D Morgan-fingerprint Tanimoto to the active) -- decoys
should plausibly fit the same binding pocket by bulk/polarity while not
being trivial close analogs of the real drug.

This is NOT literal DUD-E data (DUD-E has no beta-lactamase/PBP targets
covering this project's specific 8 receptors -- checked directly, see
README/DECISIONS_AND_LIMITATIONS.md for the citation trail) and it does
not reproduce DUD-E's full pipeline (no ZINC lead-like source library, no
LADS/analog-bias correction). It reuses the two things that actually
matter for this project's honesty bar: physicochemical matching + 2D
topological dissimilarity, sourced from real PubChem compounds rather
than synthetic/fabricated structures.

Usage:
  python3 generate_decoys.py --target shv1_tazobactam --active-smiles "..." \
      --n-decoys 30 --out-dir results/docking/accuracy_benchmark/shv1_tazobactam
"""
import argparse
import json
import random
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Crippen, Descriptors, rdMolDescriptors
from rdkit.DataStructs import TanimotoSimilarity

RDLogger.DisableLog("rdApp.*")

PUG_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
EUTILS_ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
ALLOWED_ELEMENTS = {"C", "H", "N", "O", "S", "P", "F", "Cl", "Br"}
RANDOM_SEED = 42
TANIMOTO_MAX = 0.35          # decoy must be this dissimilar (or more) from the active
DECOY_DEDUP_MAX = 0.90       # decoys must be at least this dissimilar from each other
BATCH_SIZE = 40              # CIDs per PUG-REST property batch fetch
MAX_CANDIDATES_PER_WINDOW = 600
HTTP_SLEEP = 0.34            # stay under NCBI's ~3 req/s unauthenticated rate limit


def _http_get(url: str, retries: int = 3) -> bytes:
    last_err = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                data = resp.read()
            time.sleep(HTTP_SLEEP)
            return data
        except Exception as e:
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"GET failed after {retries} retries: {url}\n{last_err}")


def esearch_cids(mw_lo, mw_hi, logp_lo, logp_hi, retmax=MAX_CANDIDATES_PER_WINDOW):
    term = f"{mw_lo:.2f}:{mw_hi:.2f}[MolecularWeight] AND {logp_lo:.2f}:{logp_hi:.2f}[XLogP]"
    url = EUTILS_ESEARCH + "?" + urllib.parse.urlencode(
        {"db": "pccompound", "term": term, "retmax": retmax, "retmode": "json"}
    )
    data = json.loads(_http_get(url))
    ids = data.get("esearchresult", {}).get("idlist", [])
    return [int(x) for x in ids]


def fetch_properties(cids):
    """Batch-fetch MolecularFormula/MolecularWeight/XLogP/IsomericSMILES/
    CovalentUnitCount for a list of CIDs. Returns {cid: {...}}, skipping
    CIDs PubChem doesn't recognize as compounds (some esearch pccompound
    hits are SID-adjacent oddities) rather than raising."""
    out = {}
    for i in range(0, len(cids), BATCH_SIZE):
        batch = cids[i:i + BATCH_SIZE]
        cid_str = ",".join(str(c) for c in batch)
        url = (f"{PUG_BASE}/compound/cid/{cid_str}/property/"
               f"MolecularFormula,MolecularWeight,XLogP,IsomericSMILES,CovalentUnitCount/JSON")
        try:
            data = json.loads(_http_get(url))
        except Exception as e:
            print(f"  [warn] property batch fetch failed ({len(batch)} CIDs): {e}", file=sys.stderr)
            continue
        for prop in data.get("PropertyTable", {}).get("Properties", []):
            out[prop["CID"]] = prop
    return out


def formula_elements_ok(formula: str) -> bool:
    import re
    elems = re.findall(r"[A-Z][a-z]?", formula)
    return all(e in ALLOWED_ELEMENTS for e in elems) and "C" in elems


def morgan_fp(mol):
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)


def embed_3d(mol, seed=RANDOM_SEED):
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    ok = AllChem.EmbedMolecule(mol, params)
    if ok != 0:
        # retry with random coords as a fallback, same convention meeko/RDKit
        # ligand prep elsewhere in this project falls back to
        params.useRandomCoords = True
        ok = AllChem.EmbedMolecule(mol, params)
        if ok != 0:
            return None
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=2000)
    except Exception:
        pass
    return mol


def generate_decoys(active_smiles: str, n_decoys: int, seed: int = RANDOM_SEED):
    active_mol = Chem.MolFromSmiles(active_smiles)
    if active_mol is None:
        raise ValueError(f"Could not parse active SMILES: {active_smiles}")
    active_fp = morgan_fp(active_mol)
    active_mw = Descriptors.MolWt(active_mol)
    active_logp = Crippen.MolLogP(active_mol)

    rng = random.Random(seed)
    accepted = []       # list of dicts
    accepted_fps = []
    seen_cids = set()
    tolerance_steps = [
        (max(15.0, 0.08 * active_mw), 1.0),
        (max(25.0, 0.15 * active_mw), 1.5),
        (max(40.0, 0.25 * active_mw), 2.5),
    ]

    window_log = []
    for step_i, (mw_tol, logp_tol) in enumerate(tolerance_steps):
        if len(accepted) >= n_decoys:
            break
        mw_lo, mw_hi = active_mw - mw_tol, active_mw + mw_tol
        logp_lo, logp_hi = active_logp - logp_tol, active_logp + logp_tol
        print(f"  window {step_i}: MW [{mw_lo:.1f},{mw_hi:.1f}]  logP [{logp_lo:.2f},{logp_hi:.2f}]")
        try:
            cids = esearch_cids(mw_lo, mw_hi, logp_lo, logp_hi)
        except Exception as e:
            print(f"  [warn] esearch failed for window {step_i}: {e}", file=sys.stderr)
            continue
        cids = [c for c in cids if c not in seen_cids]
        rng.shuffle(cids)
        window_log.append({"window": step_i, "mw_tol": mw_tol, "logp_tol": logp_tol,
                            "n_candidates_returned": len(cids)})
        print(f"    {len(cids)} new candidate CIDs")

        props = fetch_properties(cids)
        for cid in cids:
            if len(accepted) >= n_decoys:
                break
            seen_cids.add(cid)
            p = props.get(cid)
            if p is None:
                continue
            if int(p.get("CovalentUnitCount", 0) or 0) != 1:
                continue
            formula = p.get("MolecularFormula", "")
            if not formula_elements_ok(formula):
                continue
            smi = p.get("IsomericSMILES") or p.get("SMILES")
            if not smi:
                continue
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            mw = Descriptors.MolWt(mol)
            logp = Crippen.MolLogP(mol)
            if not (mw_lo - 5 <= mw <= mw_hi + 5):
                continue  # re-verify with RDKit's own MW, don't just trust PubChem's
            fp = morgan_fp(mol)
            sim_to_active = TanimotoSimilarity(active_fp, fp)
            if sim_to_active > TANIMOTO_MAX:
                continue
            if any(TanimotoSimilarity(fp, f2) > DECOY_DEDUP_MAX for f2 in accepted_fps):
                continue
            heavy = mol.GetNumHeavyAtoms()
            if heavy < 5:
                continue
            accepted.append({
                "cid": cid, "smiles": Chem.MolToSmiles(mol), "formula": formula,
                "mw_rdkit": round(mw, 2), "logp_rdkit": round(logp, 2),
                "mw_pubchem": p.get("MolecularWeight"), "xlogp_pubchem": p.get("XLogP"),
                "tanimoto_to_active": round(sim_to_active, 3),
                "window": step_i,
            })
            accepted_fps.append(fp)

    return {
        "active_smiles": Chem.MolToSmiles(active_mol),
        "active_mw": round(active_mw, 2),
        "active_logp": round(active_logp, 2),
        "n_requested": n_decoys,
        "n_found": len(accepted),
        "windows": window_log,
        "decoys": accepted,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", required=True)
    ap.add_argument("--active-smiles", required=True)
    ap.add_argument("--n-decoys", type=int, default=30)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    (out_dir / "decoy_sdf").mkdir(parents=True, exist_ok=True)

    print(f"=== {args.target}: generating {args.n_decoys} decoys ===")
    result = generate_decoys(args.active_smiles, args.n_decoys, seed=args.seed)
    print(f"  found {result['n_found']} / {result['n_requested']}")

    manifest_path = out_dir / "decoy_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  manifest: {manifest_path}")

    n_embedded = 0
    for i, d in enumerate(result["decoys"]):
        mol = Chem.MolFromSmiles(d["smiles"])
        mol3d = embed_3d(mol, seed=args.seed)
        if mol3d is None:
            print(f"  [warn] 3D embedding failed for CID {d['cid']}, skipping")
            continue
        sdf_path = out_dir / "decoy_sdf" / f"decoy_{i:02d}_cid{d['cid']}.sdf"
        writer = Chem.SDWriter(str(sdf_path))
        mol3d.SetProp("_Name", f"decoy_{i:02d}_cid{d['cid']}")
        writer.write(mol3d)
        writer.close()
        n_embedded += 1
    print(f"  3D-embedded {n_embedded} / {len(result['decoys'])} decoys -> {out_dir/'decoy_sdf'}")

    if result["n_found"] < args.n_decoys:
        print(f"  [NOTE] only found {result['n_found']} of {args.n_decoys} requested decoys "
              f"even after widening the property window {len(result['windows'])}x -- "
              f"reported honestly in the manifest, not padded.")


if __name__ == "__main__":
    main()

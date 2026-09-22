#!/usr/bin/env python3
"""
PathTwin Stage 4 — Mutation Impact Simulator
Predicts how a point mutation affects drug binding using:
  - FoldX BuildModel for mutation introduction + delta-delta-G (thermodynamic signal)
  - AutoDock Vina re-run for binding affinity comparison (docking signal)
Two independent signals on the same mutation = defensible result.
"""

import os
import re
import sys
import json
import shutil
import tempfile
import argparse
import subprocess
import yaml
from pathlib import Path

# Project root
ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"
RESULTS_DIR = ROOT / "results" / "mutation_sim"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Load tool paths
with open(CONFIG_DIR / "tools.yml") as f:
    _tools = yaml.safe_load(f)
FOLDX_BIN = Path(_tools["tools"]["foldx"]["binary"])

AA1_TO_3 = {
    "A":"ALA","C":"CYS","D":"ASP","E":"GLU","F":"PHE",
    "G":"GLY","H":"HIS","I":"ILE","K":"LYS","L":"LEU",
    "M":"MET","N":"ASN","P":"PRO","Q":"GLN","R":"ARG",
    "S":"SER","T":"THR","V":"VAL","W":"TRP","Y":"TYR"
}


def _foldx_notation(chain: str, resnum: int, wt: str, mut: str) -> str:
    """
    FoldX individual list notation: <WT><Chain><ResNum><Mut>
    e.g. chain A, Glu166 → Ala  ==>  EA166A
    """
    return f"{wt}{chain}{resnum}{mut}"


def _foldx_notation_multi(mutations: list[dict]) -> str:
    """
    Combined-mutant FoldX individual list notation: comma-separated single
    notations on one line, e.g. "SA109L,DA222N,PA225S" — FoldX applies all
    of them together in one BuildModel run, producing a single combined
    mutant (not three independent single mutants). Added for targets like
    OXA-239 (S109L+D222N+P225S), which is defined in the literature as one
    combined variant, not three separate ones to compare individually.
    """
    return ",".join(
        _foldx_notation(m["chain"], m["resnum"], m["wt"], m["mut"])
        for m in mutations
    )


def _repair_pdb(pdb_path: Path, workdir: Path) -> Path:
    """
    FoldX RepairPDB — required before BuildModel to resolve any
    missing atoms / clashes in the input structure.
    Returns path to repaired PDB.
    """
    shutil.copy(pdb_path, workdir / pdb_path.name)
    cmd = [
        str(FOLDX_BIN),
        "--command", "RepairPDB",
        "--pdb", pdb_path.name,
        "--pdb-dir", str(workdir),
        "--output-dir", str(workdir),
        "--noHeader", "1"
    ]
    result = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FoldX RepairPDB failed:\n{result.stderr}")

    repaired = workdir / f"{pdb_path.stem}_Repair.pdb"
    if not repaired.exists():
        raise FileNotFoundError(f"Expected repaired PDB not found: {repaired}")
    return repaired


def _introduce_mutation(
    pdb_path: Path,
    chain: str,
    resnum: int,
    wildtype_aa: str,
    mutant_aa: str,
    workdir: Path,
    extra_mutations: list[dict] | None = None,
) -> tuple[Path, float]:
    """
    FoldX BuildModel — introduces point mutation, returns:
      (mutant_pdb_path, ddg_kcal_mol)
    This function is the FoldX swap point: to upgrade to a newer
    engine later, replace only this function's internals.

    `extra_mutations`: optional list of additional {"chain","resnum","wt","mut"}
    dicts to apply TOGETHER with the primary mutation in one combined
    BuildModel run (one individual-list line, comma-separated) -- for
    literature variants defined as a combination of several substitutions
    (e.g. OXA-239 = S109L+D222N+P225S), not three independent single mutants.
    """
    # Write individual list file
    if extra_mutations:
        all_muts = [
            {"chain": chain, "resnum": resnum, "wt": wildtype_aa, "mut": mutant_aa}
        ] + extra_mutations
        notation = _foldx_notation_multi(all_muts)
    else:
        notation = _foldx_notation(chain, resnum, wildtype_aa, mutant_aa)
    individual_list = workdir / "individual_list.txt"
    individual_list.write_text(notation + ";\n")

    # Repair first
    repaired_pdb = _repair_pdb(pdb_path, workdir)

    # BuildModel
    cmd = [
        str(FOLDX_BIN),
        "--command", "BuildModel",
        "--pdb", repaired_pdb.name,
        "--pdb-dir", str(workdir),
        "--mutant-file", str(individual_list),
        "--output-dir", str(workdir),
        "--numberOfRuns", "3",
        "--noHeader", "1"
    ]
    result = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FoldX BuildModel failed:\n{result.stderr}")

    # Parse ΔΔG from Average_ file
    avg_files = list(workdir.glob("Average_*.fxout"))
    if not avg_files:
        raise FileNotFoundError("FoldX Average_ output file not found")
    avg_text = avg_files[0].read_text()
    # ΔΔG is in the second data column of the Average file
    ddg = None
    for line in avg_text.splitlines():
        if line.strip() and not line.startswith("Pdb"):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    ddg = float(parts[1])
                    break
                except ValueError:
                    continue
    if ddg is None:
        raise ValueError("Could not parse ΔΔG from FoldX Average file")

    # Find mutant PDB — FoldX names it <stem>_1.pdb
    mutant_pdb = workdir / f"{repaired_pdb.stem}_1.pdb"
    if not mutant_pdb.exists():
        candidates = list(workdir.glob(f"{repaired_pdb.stem}_*.pdb"))
        if not candidates:
            raise FileNotFoundError("FoldX mutant PDB not found")
        mutant_pdb = candidates[0]

    # Clean FoldX's non-spec-compliant PDB output before it goes to obabel:
    # strip the banner lines FoldX prepends before the first ATOM/HETATM
    # record, and append a proper END record (FoldX emits neither END nor
    # TER), since obabel silently rejects the raw output (exit 0, 0 atoms).
    raw_lines = mutant_pdb.read_text().splitlines()
    first_atom_idx = next(
        (i for i, line in enumerate(raw_lines)
         if line.startswith("ATOM") or line.startswith("HETATM")),
        0
    )
    cleaned_lines = raw_lines[first_atom_idx:]
    cleaned_lines.append("END")
    cleaned_pdb = workdir / f"{mutant_pdb.stem}_cleaned.pdb"
    cleaned_pdb.write_text("\n".join(cleaned_lines) + "\n")
    mutant_pdb = cleaned_pdb

    return mutant_pdb, ddg


def _prep_receptor_pdbqt(pdb_path: Path, workdir: Path) -> Path:
    """Convert mutant PDB to PDBQT using obabel for Vina."""
    pdbqt_path = workdir / (pdb_path.stem + ".pdbqt")
    cmd = [
        "obabel",
        "-ipdb", str(pdb_path),
        "-opdbqt", "-O", str(pdbqt_path),
        "--partialcharge", "gasteiger",
        "-xr"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not pdbqt_path.exists():
        raise RuntimeError(
            f"Receptor PDBQT prep failed:\n{result.stderr}"
        )
    # obabel can exit 0 while silently converting 0 molecules (e.g. it
    # considers the input PDB malformed) and still leave an empty file
    # behind. Catch that here with the obabel output attached, instead of
    # letting Vina "dock" against a receptor with no atoms downstream.
    size = pdbqt_path.stat().st_size
    has_atoms = size > 0 and re.search(
        r"^(ATOM|HETATM)", pdbqt_path.read_text(errors="replace"), re.MULTILINE
    )
    if not has_atoms:
        raise RuntimeError(
            f"Receptor PDBQT prep produced an empty/invalid file for "
            f"{pdb_path} (obabel exited 0 but wrote no atoms -- size={size} bytes).\n"
            f"obabel stdout:\n{result.stdout}\n"
            f"obabel stderr:\n{result.stderr}"
        )
    return pdbqt_path

def _run_vina(receptor_pdbqt: Path, ligand_pdbqt: Path,
              box: dict, workdir: Path, label: str = "") -> float:
    """Run AutoDock Vina, return top binding affinity (kcal/mol)."""
    tag = label or "vina"

    # ── Pre-flight: make sure the inputs Vina is about to read are real ───
    # A missing or empty receptor doesn't make Vina error out -- it happily
    # "docks" against zero atoms and reports a bogus ~0.000 affinity that
    # parses just fine, which is exactly how this failed silently before.
    if not receptor_pdbqt.exists():
        raise FileNotFoundError(
            f"[{tag}] Receptor PDBQT does not exist: {receptor_pdbqt}"
        )
    receptor_size = receptor_pdbqt.stat().st_size
    if receptor_size == 0:
        raise ValueError(
            f"[{tag}] Receptor PDBQT is empty (0 bytes): {receptor_pdbqt}"
        )
    if not ligand_pdbqt.exists():
        raise FileNotFoundError(
            f"[{tag}] Ligand PDBQT does not exist: {ligand_pdbqt}"
        )
    print(f"  [debug] {tag} receptor: {receptor_pdbqt} ({receptor_size} bytes)")
    print(f"  [debug] {tag} ligand  : {ligand_pdbqt}")

    workdir.mkdir(parents=True, exist_ok=True)
    out_path = workdir / "docked.pdbqt"
    cmd = [
        "vina",
        "--receptor", str(receptor_pdbqt),
        "--ligand", str(ligand_pdbqt),
        "--center_x", str(box["center_x"]),
        "--center_y", str(box["center_y"]),
        "--center_z", str(box["center_z"]),
        "--size_x", str(box["size_x"]),
        "--size_y", str(box["size_y"]),
        "--size_z", str(box["size_z"]),
        "--exhaustiveness", str(box.get("exhaustiveness", 32)),
        "--seed", str(box.get("random_seed", 42)),
        "--out", str(out_path),
        "--num_modes", "9"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    # Vina 1.2.x writes scores to stdout
    output = result.stdout + result.stderr

    # ── Debug dump: full raw Vina stdout+stderr, always, labeled by run ───
    # This is the piece that was previously invisible: a "successful" run
    # (exit 0, a parseable number) could still be scoring garbage input.
    print(f"\n  ----- Vina raw output [{tag}] -----")
    print(f"  cmd: {' '.join(cmd)}")
    print(f"  returncode: {result.returncode}")
    print(output.rstrip() if output.strip() else "  (no stdout/stderr captured)")
    print(f"  ----- end Vina raw output [{tag}] -----\n")

    if result.returncode != 0:
        raise RuntimeError(f"[{tag}] Vina exited with code {result.returncode}:\n{output}")

    scores = re.findall(r"^\s+1\s+([-\d.]+)", output, re.MULTILINE)
    if not scores:
        # fallback: grab first numeric score line
        scores = re.findall(r"^\s+\d+\s+([-\d.]+)", output, re.MULTILINE)
    if not scores:
        raise ValueError(f"[{tag}] Could not parse Vina score from output:\n{output}")

    score = float(scores[0])
    if score == 0.0:
        print(
            f"  [warning] Parsed Vina score for [{tag}] is exactly 0.000 -- "
            f"real binding affinities essentially never land exactly on zero. "
            f"This usually means the receptor had no atoms (empty/invalid "
            f"PDBQT), not a genuine docking result. Check the receptor prep "
            f"step for {tag}."
        )
    return score


def predict_mutation_impact(
    receptor_pdb: str,
    ligand_pdbqt: str,
    chain: str,
    residue_num: int,
    wildtype_aa: str,
    mutant_aa: str,
    binding_box_json: str,
    extra_mutations: list[dict] | None = None,
    label_override: str | None = None,
) -> dict:
    """
    Core PathTwin mutation impact function.
    Returns dict with both thermodynamic (FoldX) and docking (Vina) signals.

    `extra_mutations`: see _introduce_mutation -- applies all mutations
    together as ONE combined FoldX mutant (e.g. OXA-239's S109L+D222N+P225S),
    not three independent single-mutant comparisons.
    `label_override`: use this as the mutation_label / result workdir name
    instead of the auto-generated single-mutation label (needed for combined
    mutants so the workdir/result reflects all substitutions, not just the
    primary one).
    """
    receptor_pdb = Path(receptor_pdb)
    ligand_pdbqt = Path(ligand_pdbqt)
    box_config = Path(binding_box_json)

    with open(box_config) as f:
        box = json.load(f)

    if label_override:
        mutation_label = label_override
    elif extra_mutations:
        all_muts = [
            {"chain": chain, "resnum": residue_num, "wt": wildtype_aa, "mut": mutant_aa}
        ] + extra_mutations
        mutation_label = "+".join(f"{m['wt']}{m['chain']}{m['resnum']}{m['mut']}" for m in all_muts)
    else:
        mutation_label = f"{chain}:{residue_num}:{wildtype_aa}:{mutant_aa}"
    workdir = RESULTS_DIR / mutation_label.replace(":", "_").replace("+", "_")
    workdir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"PathTwin Mutation Simulator")
    print(f"Mutation : {mutation_label}")
    print(f"Receptor : {receptor_pdb.name}")
    print(f"Ligand   : {ligand_pdbqt.name}")
    print(f"{'='*60}")

    # ── Wildtype Vina score (use existing prepped receptor if available) ──
    wt_receptor_pdbqt = receptor_pdb.parent / (receptor_pdb.stem + "_receptor.pdbqt")
    if not wt_receptor_pdbqt.exists():
        print("Prepping wildtype receptor PDBQT...")
        wt_receptor_pdbqt = _prep_receptor_pdbqt(receptor_pdb, workdir)

    print("Running Vina on wildtype receptor...")
    wt_score = _run_vina(wt_receptor_pdbqt, ligand_pdbqt, box, workdir / "wt_vina", label="wildtype")
    print(f"  Wildtype Vina score : {wt_score:.3f} kcal/mol")

    # ── Introduce mutation via FoldX ──────────────────────────────────────
    print(f"\nIntroducing mutation {mutation_label} via FoldX BuildModel...")
    mutant_pdb, ddg = _introduce_mutation(
        receptor_pdb, chain, residue_num, wildtype_aa, mutant_aa, workdir,
        extra_mutations=extra_mutations,
    )
    print(f"  FoldX ΔΔG          : {ddg:+.3f} kcal/mol")
    print(f"  (positive = destabilizing = potential resistance mechanism)")

    # ── Prep mutant receptor + re-run Vina ───────────────────────────────
    print("\nPrepping mutant receptor PDBQT...")
    mutant_pdbqt = _prep_receptor_pdbqt(mutant_pdb, workdir)

    # ── Explicit check: is the mutant receptor PDBQT actually there and
    # non-empty before we hand it to Vina? _prep_receptor_pdbqt already
    # validates this, but confirm again here at the call site so a wrong
    # path getting passed in is caught with a clear message too, not a
    # silent ~0.000 downstream.
    if not mutant_pdbqt.exists():
        raise FileNotFoundError(
            f"Mutant receptor PDBQT missing before Vina run: {mutant_pdbqt}"
        )
    if mutant_pdbqt.stat().st_size == 0:
        raise ValueError(
            f"Mutant receptor PDBQT is empty before Vina run: {mutant_pdbqt}"
        )
    print(f"  Mutant receptor PDBQT : {mutant_pdbqt} ({mutant_pdbqt.stat().st_size} bytes)")

    print("Running Vina on mutant receptor...")
    (workdir / "mut_vina").mkdir(exist_ok=True)
    mut_score = _run_vina(mutant_pdbqt, ligand_pdbqt, box, workdir / "mut_vina", label="mutant")
    print(f"  Mutant Vina score  : {mut_score:.3f} kcal/mol")

    delta = mut_score - wt_score
    print(f"\n  Delta affinity     : {delta:+.3f} kcal/mol")
    print(f"  (positive = weaker binding in mutant = resistance signal)")
    print(f"  FoldX ΔΔG          : {ddg:+.3f} kcal/mol")

    if delta > 0 and ddg > 0:
        interpretation = "BOTH signals agree: mutation likely reduces drug binding (resistance)"
    elif delta > 0 and ddg <= 0:
        interpretation = "Vina suggests weaker binding; FoldX shows stabilizing — ambiguous"
    elif delta <= 0 and ddg > 0:
        interpretation = "FoldX destabilizing; Vina shows tighter binding — ambiguous"
    else:
        interpretation = "Both signals: mutation does not reduce drug binding"
    print(f"  Interpretation     : {interpretation}")

    result = {
        "mutation_label": mutation_label,
        "wildtype_vina_score": wt_score,
        "mutant_vina_score": mut_score,
        "delta_affinity": round(delta, 3),
        "ddg_foldx": round(ddg, 3),
        "mutant_structure_path": str(mutant_pdb),
        "interpretation": interpretation
    }

    # Save result JSON
    out_json = workdir / "result.json"
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResult saved to: {out_json}")

    return result


def _parse_stage3_mutations(stage3_tab: str) -> list[dict]:
    """Parse snippy .tab output for resistance-relevant variants."""
    mutations = []
    with open(stage3_tab) as f:
        for line in f:
            if line.startswith("CHROM") or not line.strip():
                continue
            parts = line.strip().split("\t")
            if len(parts) < 11:
                continue
            # snippy tab: CHROM POS TYPE REF ALT EVIDENCE FTYPE STRAND NT_POS AA_POS EFFECT GENE ...
            try:
                gene = parts[11] if len(parts) > 11 else "unknown"
                aa_change = parts[10] if len(parts) > 10 else ""
                # Parse p.Ser83Tyr style AA change
                match = re.match(r"p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})", aa_change)
                if match:
                    aa3_wt, pos, aa3_mut = match.groups()
                    # Convert 3-letter to 1-letter
                    aa3_to_1 = {v: k for k, v in AA1_TO_3.items()}
                    wt1 = aa3_to_1.get(aa3_wt.upper())
                    mut1 = aa3_to_1.get(aa3_mut.upper())
                    if wt1 and mut1:
                        mutations.append({
                            "gene": gene,
                            "chain": "A",  # default; user can override
                            "resnum": int(pos),
                            "wt": wt1,
                            "mut": mut1,
                            "aa_change": aa_change
                        })
            except (IndexError, ValueError):
                continue
    return mutations


def main():
    parser = argparse.ArgumentParser(
        description="PathTwin Stage 4: Mutation Impact Simulator"
    )
    parser.add_argument("--receptor", required=True,
                        help="Path to wildtype receptor PDB")
    parser.add_argument("--ligand", required=True,
                        help="Path to ligand PDBQT")
    parser.add_argument("--box-config", required=True,
                        help="Path to binding box JSON")
    parser.add_argument("--mutation",
                        help="Single mutation: CHAIN:RESNUM:WT:MUT (e.g. A:166:E:A)")
    parser.add_argument("--stage3-output",
                        help="Path to snippy resistance_relevant_snps.tab")
    parser.add_argument("--multi-mutation",
                        help=(
                            "Combined mutant: comma-separated CHAIN:RESNUM:WT:MUT entries, "
                            "applied TOGETHER as one FoldX BuildModel run (one combined "
                            "mutant, not independent single mutants) -- e.g. for a "
                            "literature variant defined as a set of substitutions together "
                            "(OXA-239 = S109L+D222N+P225S): "
                            "'A:109:S:L,A:222:D:N,A:225:P:S'"
                        ))
    args = parser.parse_args()

    if not args.mutation and not args.stage3_output and not args.multi_mutation:
        parser.error("Provide --mutation, --multi-mutation, or --stage3-output")

    mutations_to_run = []
    multi_mutation_set = None

    if args.mutation:
        parts = args.mutation.split(":")
        if len(parts) != 4:
            sys.exit("--mutation must be CHAIN:RESNUM:WT:MUT e.g. A:166:E:A")
        mutations_to_run.append({
            "chain": parts[0],
            "resnum": int(parts[1]),
            "wt": parts[2],
            "mut": parts[3]
        })

    if args.multi_mutation:
        entries = []
        for token in args.multi_mutation.split(","):
            parts = token.split(":")
            if len(parts) != 4:
                sys.exit(f"--multi-mutation entries must be CHAIN:RESNUM:WT:MUT, got: {token!r}")
            entries.append({
                "chain": parts[0],
                "resnum": int(parts[1]),
                "wt": parts[2],
                "mut": parts[3],
            })
        if len(entries) < 2:
            sys.exit("--multi-mutation needs at least 2 comma-separated entries "
                      "(use --mutation for a single substitution)")
        multi_mutation_set = entries

    if args.stage3_output:
        stage3_muts = _parse_stage3_mutations(args.stage3_output)
        print(f"Loaded {len(stage3_muts)} mutations from Stage 3 output")
        mutations_to_run.extend(stage3_muts)

    if multi_mutation_set:
        primary, extras = multi_mutation_set[0], multi_mutation_set[1:]
        label = "+".join(f"{m['wt']}{m['chain']}{m['resnum']}{m['mut']}" for m in multi_mutation_set)
        try:
            result = predict_mutation_impact(
                receptor_pdb=args.receptor,
                ligand_pdbqt=args.ligand,
                chain=primary["chain"],
                residue_num=primary["resnum"],
                wildtype_aa=primary["wt"],
                mutant_aa=primary["mut"],
                binding_box_json=args.box_config,
                extra_mutations=extras,
                label_override=label,
            )
            print(f"\n{'='*60}\nCOMBINED MUTANT RESULT: {label}\n{'='*60}")
            print(f"  FoldX ddG   : {result['ddg_foldx']:+.3f} kcal/mol")
            print(f"  Delta Vina  : {result['delta_affinity']:+.3f} kcal/mol")
            print(f"  {result['interpretation']}")
        except Exception as e:
            sys.exit(f"ERROR processing combined mutation {label}: {e}")
        return

    if not mutations_to_run:
        sys.exit("No mutations to process")

    all_results = []
    for m in mutations_to_run:
        try:
            result = predict_mutation_impact(
                receptor_pdb=args.receptor,
                ligand_pdbqt=args.ligand,
                chain=m["chain"],
                residue_num=m["resnum"],
                wildtype_aa=m["wt"],
                mutant_aa=m["mut"],
                binding_box_json=args.box_config
            )
            all_results.append(result)
        except Exception as e:
            print(f"ERROR processing mutation {m}: {e}", file=sys.stderr)
            continue

    # Summary table
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Mutation':<20} {'ΔΔG FoldX':>12} {'ΔVina':>10} {'Interpretation'}")
    print("-"*80)
    for r in all_results:
        print(
            f"{r['mutation_label']:<20} "
            f"{r['ddg_foldx']:>+12.3f} "
            f"{r['delta_affinity']:>+10.3f}  "
            f"{r['interpretation']}"
        )


if __name__ == "__main__":
    main()

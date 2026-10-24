#!/usr/bin/env python3
"""
PathTwin Stage 2 -- generic, dependency-free docking-pose validation
analysis (distance from the top Vina pose to named pocket residues and to
a real crystallographic ligand). Written for the PBP2a/PPDHMP wet-lab
candidate run since no reusable version of this analysis existed on disk
from the original 8 Stage 2 targets (each was analyzed via one-off
interactive commands) -- built generically enough to reuse for any future
target without needing biopython (not installed in pathtwin-docking).

Usage:
  python3 analyze_docking_pose.py --pose docked.pdbqt --receptor-pdb chainA_clean.pdb \
      --pocket-residues "ARG:151,GLU:239,SER:240,ARG:241,VAL:256,VAL:277,HIS:293" \
      --crystal-ligand crystal_ligand.pdb --extra-residue "SER:403:OG"
"""
import argparse
import json
import math


def parse_pdb_atoms(path):
    """Returns list of dicts: resname, chain, resseq, atomname, x, y, z."""
    atoms = []
    with open(path) as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            atomname = line[12:16].strip()
            resname = line[17:20].strip()
            chain = line[21].strip()
            resseq = int(line[22:26].strip())
            x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            atoms.append({"resname": resname, "chain": chain, "resseq": resseq,
                          "atomname": atomname, "xyz": (x, y, z)})
    return atoms


def parse_pdbqt_top_pose(path):
    """Returns list of (x, y, z) for the first MODEL's ATOM/HETATM lines."""
    coords = []
    in_first_model = True
    with open(path) as f:
        for line in f:
            if line.startswith("ENDMDL"):
                break
            if line.startswith("ATOM") or line.startswith("HETATM"):
                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                coords.append((x, y, z))
    return coords


def dist(a, b):
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def centroid(coords):
    n = len(coords)
    return tuple(sum(c[i] for c in coords) / n for i in range(3))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pose", required=True, help="Vina output PDBQT (top pose = first MODEL)")
    ap.add_argument("--receptor-pdb", required=True, help="Receptor PDB with named residues (e.g. chainA_clean.pdb)")
    ap.add_argument("--pocket-residues", required=True,
                     help="Comma-separated RESNAME:RESSEQ list, CA atom used for each, e.g. 'ARG:151,GLU:239'")
    ap.add_argument("--crystal-ligand", default=None, help="PDB file with the real crystal ligand's HETATM records")
    ap.add_argument("--extra-residue", action="append", default=[],
                     help="RESNAME:RESSEQ:ATOMNAME, reported as an extra named distance (e.g. catalytic Ser)")
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    pose_coords = parse_pdbqt_top_pose(args.pose)
    pose_centroid = centroid(pose_coords)
    receptor_atoms = parse_pdb_atoms(args.receptor_pdb)

    result = {"n_pose_atoms": len(pose_coords), "pose_centroid": pose_centroid}

    pocket_distances = {}
    for spec in args.pocket_residues.split(","):
        resname, resseq = spec.split(":")
        resseq = int(resseq)
        ca = next((a for a in receptor_atoms if a["resname"] == resname and a["resseq"] == resseq and a["atomname"] == "CA"), None)
        if ca is None:
            pocket_distances[f"{resname}{resseq}"] = None
            continue
        min_d = min(dist(c, ca["xyz"]) for c in pose_coords)
        pocket_distances[f"top_pose_distance_to_{resname.title()}{resseq}_CA_A"] = round(min_d, 2)
    result["pocket_residue_distances_A"] = pocket_distances

    for spec in args.extra_residue:
        resname, resseq, atomname = spec.split(":")
        resseq = int(resseq)
        atom = next((a for a in receptor_atoms if a["resname"] == resname and a["resseq"] == resseq and a["atomname"] == atomname), None)
        if atom is None:
            continue
        min_d = min(dist(c, atom["xyz"]) for c in pose_coords)
        result[f"distance_to_{resname.title()}{resseq}_{atomname}_A"] = round(min_d, 2)

    if args.crystal_ligand:
        crystal_atoms = parse_pdb_atoms(args.crystal_ligand)
        crystal_coords = [a["xyz"] for a in crystal_atoms]
        crystal_centroid = centroid(crystal_coords)
        closest = min(dist(pc, cc) for pc in pose_coords for cc in crystal_coords)
        result["closest_atom_to_crystal_ligand_A"] = round(closest, 2)
        result["centroid_to_centroid_vs_crystal_A"] = round(dist(pose_centroid, crystal_centroid), 2)

    print(json.dumps(result, indent=2))
    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import numpy as np


def parse_pdb_hetatm_coords(path, resname=None):
    coords = []
    for line in open(path):
        if line.startswith("HETATM") or line.startswith("ATOM"):
            rn = line[17:20].strip()
            if resname and rn != resname:
                continue
            x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            elem = line[76:78].strip() or line[12:14].strip()
            if elem.upper().startswith("H"):
                continue
            coords.append((x, y, z))
    return np.array(coords)


def parse_pdbqt_pose_coords(path, model_idx=0):
    coords = []
    in_model = (model_idx == 0)
    current_model = 0
    for line in open(path):
        if line.startswith("MODEL"):
            current_model = int(line.split()[1]) - 1
            in_model = (current_model == model_idx)
        if in_model and (line.startswith("ATOM") or line.startswith("HETATM")):
            x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            atype = line[77:79].strip() if len(line) > 78 else ""
            if atype in ("H", "HD"):
                continue
            coords.append((x, y, z))
        if line.startswith("ENDMDL") and in_model:
            break
    return np.array(coords)


crystal = parse_pdb_hetatm_coords(
    "/home/kell/PathTwin/data/raw/structures/1VM1_TAZ_A504_aligned_to_1SHV.pdb", resname="TAZ")
docked = parse_pdbqt_pose_coords(
    "/home/kell/PathTwin/results/docking/accuracy_benchmark/shv1_tazobactam_corrected/docked/active_tazobactam_docked.pdbqt",
    model_idx=0)

print("crystal TAZ heavy atoms:", len(crystal))
print("docked pose 1 heavy atoms:", len(docked))

dmat = np.linalg.norm(crystal[:, None, :] - docked[None, :, :], axis=-1)
print(f"closest atom-to-atom distance: {dmat.min():.3f} A")

c1 = crystal.mean(axis=0)
c2 = docked.mean(axis=0)
print(f"centroid-to-centroid distance: {np.linalg.norm(c1 - c2):.3f} A")

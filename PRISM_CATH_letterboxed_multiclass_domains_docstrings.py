#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Resumable multiclass YOLOv26 training pipeline for domain detection.

Skips expensive steps when intermediates already exist and are up-to-date:
- Re-uses existing PNG contact maps if newer than source PDB/CIF (unless --force_build).
- Re-uses existing YOLO label .txt files if newer than the annotations CSV (unless --force_build).
- Re-uses existing data.yaml unless --force_build.
- Can resume YOLO training if a prior run folder exists (see --resume_train).

Usage (example):
  conda activate multiqc
  python train_adhesin_yolo_from_annotations_resumable.py \
    --csv /oceanstor/scratch/tllseedorf/e1103389/adhesin/shuan_domains_with_filenames_and_found_files_updated.csv \
    --pdb_root test/data \
    --out /oceanstor/scratch/tllseedorf/e1103389/adhesin/coco/datasets/adhesin_v1 \
    --px_per_res 5 --dpi 100 --val_frac 0.15 \
    --epochs 100 --batch 8 --weights yolov26n.pt \
    --resume_train

Notes:
- CSV must include domain_start, domain_end, domain_type, and protein_id columns.
- Domain classes are discovered from all non-empty domain_type values in the annotation file.
- Bounding boxes are diagonal squares mapping [start,end] residue ranges.
"""

from collections import Counter


import os, csv, math, random, argparse, shutil, sys
from collections import defaultdict
import pandas as pd
import numpy as np
import yaml
# ---------- Multi-class labels ----------
# Domain classes are discovered from the annotation file.
# The generated integer IDs must stay consistent with data.yaml "names".
CLASS_NAMES = []
def project_has_pt_file(project_dir):
    """Return True if any .pt file exists recursively inside project_dir."""
    if not os.path.isdir(project_dir):
        return False

    for root, _, files in os.walk(project_dir):
        for f in files:
            if f.endswith(".pt"):
                return True

    return False

def _latest_mtime_recursive(path):
    """Return the newest mtime inside path, or the directory mtime if empty."""
    if not os.path.isdir(path):
        return -1

    newest = os.path.getmtime(path)
    for root, dirs, files in os.walk(path):
        for name in dirs + files:
            item = os.path.join(root, name)
            try:
                newest = max(newest, os.path.getmtime(item))
            except OSError:
                pass
    return newest

def find_model_in_run(run_dir):
    """Find best.pt first, else last.pt, inside a YOLO run directory."""
    if not os.path.isdir(run_dir):
        return None

    preferred = [
        os.path.join(run_dir, "weights", "best.pt"),
        os.path.join(run_dir, "weights", "last.pt"),
    ]
    for path in preferred:
        if os.path.isfile(path):
            return path

    best_candidates = []
    last_candidates = []
    for root, _, files in os.walk(run_dir):
        for filename in files:
            path = os.path.join(root, filename)
            if filename == "best.pt":
                best_candidates.append(path)
            elif filename == "last.pt":
                last_candidates.append(path)

    if best_candidates:
        return max(best_candidates, key=os.path.getmtime)
    if last_candidates:
        return max(last_candidates, key=os.path.getmtime)
    return None

def find_latest_run_with_weights(project_dir):
    """Return the latest modified direct child run directory containing best.pt or last.pt."""
    if not os.path.isdir(project_dir):
        return None

    candidates = []
    for name in os.listdir(project_dir):
        run_dir = os.path.join(project_dir, name)
        if not os.path.isdir(run_dir):
            continue
        model_path = find_model_in_run(run_dir)
        if model_path is None:
            continue
        candidates.append((name, _latest_mtime_recursive(run_dir)))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0][0]
def normalize_domain_type(domain_type: str) -> str | None:
    """Normalize a domain label from the annotation file into a stable class name."""
    if domain_type is None:
        return None

    value = str(domain_type).strip()
    if not value or value.lower() in {"nan", "none", "null"}:
        return None

    return value

def discover_domain_classes(df):
    """Return sorted domain classes discovered from the annotation dataframe."""
    if "domain_type" not in df.columns:
        raise RuntimeError("Annotation file must contain a domain_type column for multiclass training.")

    classes = []
    seen = set()
    for value in df["domain_type"]:
        cname = normalize_domain_type(value)
        if cname is None:
            continue
        key = cname.lower()
        if key in seen:
            continue
        seen.add(key)
        classes.append(cname)

    classes = sorted(classes, key=lambda x: x.lower())
    if not classes:
        raise RuntimeError("No usable domain classes were found in the annotation file.")

    return classes

def get_class_id(domain_type: str, class_to_id: dict):
    """Return the numeric class ID for a domain type, or None if the label is empty or unknown."""
    cname = normalize_domain_type(domain_type)
    if cname is None:
        return None
    return class_to_id.get(cname.lower())
# ---------- Light parsers / IO ----------
def count_ca_atoms(pdb_file):
    """Count alpha-carbon atoms in a PDB or mmCIF structure file, using Bio.PDB with a raw PDB fallback."""
    from Bio.PDB import PDBParser, MMCIFParser

    try:
        if pdb_file.endswith(".pdb"):
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure("prot", pdb_file)
        elif pdb_file.endswith(".cif"):
            parser = MMCIFParser(QUIET=True)
            structure = parser.get_structure("prot", pdb_file)
        else:
            raise ValueError("Unsupported file format: " + pdb_file)

        n = 0
        for model in structure:
            for chain in model:
                for res in chain:
                    if "CA" in res:
                        n += 1

        if n > 0:
            return n

    except Exception as e:
        print(f"[WARN] Bio.PDB failed for {pdb_file}: {e}")

    # Fallback for nonstandard PDB files
    n = 0
    with open(pdb_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                atom_name = line[12:16].strip()
                if atom_name == "CA":
                    n += 1

    if n == 0:
        raise RuntimeError(f"No CA atoms found in: {pdb_file}")

    return n

def pdb_to_ca_distance_matrix(pdb_file):
    """Full distance matrix with fallback parser for fixed-column PDB and whitespace-delimited ATOM files."""
    from Bio.PDB import PDBParser, MMCIFParser
    import numpy as np

    ca_coords = []

    try:
        if pdb_file.endswith(".pdb"):
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure("prot", pdb_file)
        elif pdb_file.endswith(".cif"):
            parser = MMCIFParser(QUIET=True)
            structure = parser.get_structure("prot", pdb_file)
        else:
            raise ValueError("Unsupported file format: " + pdb_file)

        for model in structure:
            for chain in model:
                for res in chain:
                    if "CA" in res:
                        ca_coords.append(res["CA"].get_coord())

    except Exception as e:
        print(f"[WARN] Bio.PDB failed for {pdb_file}: {e}")

    if len(ca_coords) == 0:
        print(f"[INFO] Falling back to raw ATOM parsing for {pdb_file}")

        with open(pdb_file, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if not line.startswith(("ATOM", "HETATM")):
                    continue

                parsed = False

                # Fallback 1: classic fixed-column PDB
                try:
                    atom_name = line[12:16].strip()

                    if atom_name == "CA":
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])

                        ca_coords.append([x, y, z])
                        parsed = True

                except Exception:
                    parsed = False

                if parsed:
                    continue

                # Fallback 2: whitespace-delimited mmCIF-like ATOM format
                parts = line.split()

                if len(parts) < 13:
                    continue

                try:
                    atom_name = parts[3]

                    if atom_name != "CA":
                        continue

                    x = float(parts[10])
                    y = float(parts[11])
                    z = float(parts[12])

                    ca_coords.append([x, y, z])

                except Exception:
                    continue

    if len(ca_coords) == 0:
        raise RuntimeError(f"No CA atoms found in: {pdb_file}")

    ca_coords = np.asarray(ca_coords, dtype=float)

    N = ca_coords.shape[0]

    sq = np.sum(ca_coords**2, axis=1, keepdims=True)
    dist2 = sq + sq.T - 2 * (ca_coords @ ca_coords.T)
    dist2[dist2 < 0] = 0

    D = np.sqrt(dist2)

    return D, N

def save_distance_png_letterboxed(D, out_png, canvas_n, px_per_res=5, dpi=100):
    """Save a CA-distance matrix as a square letterboxed PNG image and return the residue-padding offset."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = D.shape[0]

    if n > canvas_n:
        raise ValueError(f"Protein length {n} is larger than canvas size {canvas_n}")

    pad_before = (canvas_n - n) // 2

    pad_value = float(np.nanmax(D)) if D.size else 0.0

    canvas = np.full(
        (canvas_n, canvas_n),
        pad_value,
        dtype=float
    )

    canvas[
        pad_before:pad_before + n,
        pad_before:pad_before + n
    ] = D

    figsize = (canvas_n * px_per_res / dpi, canvas_n * px_per_res / dpi)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.imshow(canvas, origin="lower", cmap="viridis", interpolation="nearest")
    ax.set_axis_off()
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    return pad_before
def residue_box_to_yolo(start_res, end_res, protein_n, canvas_n):
    """Convert a residue interval into normalized YOLO bounding-box coordinates on the letterboxed canvas."""
    pad_before = (canvas_n - protein_n) // 2

    x1 = pad_before + max(0, min(protein_n, int(start_res) - 1))
    y1 = pad_before + max(0, min(protein_n, int(start_res) - 1))

    x2 = pad_before + max(0, min(protein_n, int(end_res)))
    y2 = pad_before + max(0, min(protein_n, int(end_res)))

    if x2 <= x1 or y2 <= y1:
        return None

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = x2 - x1
    h = y2 - y1

    cy = canvas_n - cy

    return cx / canvas_n, cy / canvas_n, w / canvas_n, h / canvas_n
# ---------- Utils ----------
def find_max_ca_atoms(pdb_root, protein_ids):
    """Find the longest available structure among protein IDs and return its CA count plus per-protein counts."""
    max_n = 0
    file_to_n = {}

    pdb_files = [
        f for f in os.listdir(pdb_root)
        if f.endswith(".pdb") or f.endswith(".cif")
    ]

    for protein_id in protein_ids:
        protein_id = str(protein_id)

        #find {protein_id}.pdb or {protein_id}.cif in pdb_root
        pdb_file = f"{protein_id}.pdb"
        pdb_file = os.path.join(pdb_root, pdb_file)
        if not os.path.isfile(pdb_file):
            pdb_file = os.path.join(pdb_root, f"{protein_id}.cif")
            if not os.path.isfile(pdb_file):
                print(f"[WARN] Could not find PDB/CIF for {protein_id} (tried {protein_id}.pdb and {protein_id}.cif); skipping")
                continue
        else:
            print(f"[INFO] Found structure for {protein_id}: {pdb_file}")

        try:
            n = count_ca_atoms(pdb_file)
        except Exception as e:
            print(f"[WARN] Could not count CA atoms for protein_id={protein_id} using {pdb_file}: {e}")
            continue

        file_to_n[protein_id] = n
        max_n = max(max_n, n)

        print(f"[INFO] Found structure for protein_id={protein_id}: {pdb_file} ({n} CA atoms)")

    if max_n == 0:
        raise RuntimeError("Could not determine max protein length from PDB/CIF files using protein_id matching.")

    return max_n, file_to_n
def newer_than(path_a, path_b):
    """Return True if path_a exists and is newer than path_b (which must exist)."""
    if not os.path.isfile(path_a):
        return False
    if not os.path.isfile(path_b):
        return True
    return os.path.getmtime(path_a) >= os.path.getmtime(path_b)
def load_and_convert_annotation(input_csv_path):
    """Load an annotation CSV/TSV and convert supported alternate schemas into the pipeline column layout."""
    try:
        df = pd.read_csv(input_csv_path, sep="\t")
        if len(df.columns) == 1:
            df = pd.read_csv(input_csv_path, sep=",")
    except Exception:
        df = pd.read_csv(input_csv_path, sep=",")
    if "domain_id" in df.columns and "superfamily_name" in df.columns:
        print("[INFO] Detected Family Proteinase Layout. Converting...")
        df_pipeline = pd.DataFrame()
        df_pipeline["filename"] = df["domain_id"].astype(str) + ".pdb"
        df_pipeline["domain_start"] = df["start"]
        df_pipeline["domain_end"] = df["end"]
        df_pipeline["domain_type"] = df["superfamily_name"]
        df_pipeline["protein_id"] = df["domain_id"]
        return df_pipeline
    return df

def build_dataset(csv_path,
                  pdb_root,
                  out_dir,
                  px_per_res=5,
                  dpi=100,
                  val_frac=0.15,
                  seed=1337,
                  force_build=False,
                  resume=True,
                  train_files=None,
                  val_files=None):
    # Normalize paths to avoid Ultralytics datasets_dir rewriting for relative paths
    """Build or reuse a YOLO dataset from domain annotations and PDB/CIF structures, then return the data.yaml path."""
    out_dir = os.path.abspath(out_dir)
    pdb_root = os.path.abspath(pdb_root)
    csv_path = os.path.abspath(csv_path)
    df = load_and_convert_annotation(csv_path)
    class_names = discover_domain_classes(df)
    class_to_id = {name.lower(): i for i, name in enumerate(class_names)}
    print(f"[INFO] Discovered {len(class_names)} domain classes from annotations:")
    for i, name in enumerate(class_names):
        print(f"  {i}: {name}")

    groups = df.groupby("protein_id")

    # first, count the number of pdb/cif files available in the pdb_root
    avail_files = set()
    for file in os.listdir(pdb_root):
        if not file.endswith(".pdb") and not file.endswith(".cif"):
            continue
        avail_files.add(file)
    print(f"[INFO] Found {len(avail_files)} PDB/CIF files in {pdb_root}")

    images_dir = os.path.join(out_dir, "images")
    labels_dir = os.path.join(out_dir, "labels")
    for split in ("train", "val"):
        os.makedirs(os.path.join(images_dir, split), exist_ok=True)
        os.makedirs(os.path.join(labels_dir, split), exist_ok=True)

    # Split per filename into train/val
    files = sorted(list(groups.groups.keys()))
    print("filesee:", files)
    canvas_n, file_to_n = find_max_ca_atoms(pdb_root, files)
    print(f"[INFO] Longest protein length: {canvas_n} residues")
    print(f"[INFO] Standardizing all contact maps to canvas size: {canvas_n} x {canvas_n}")
    if train_files is not None and val_files is not None:
        val_set = set(val_files)
        train_set = set(train_files)
    else:
        random.Random(seed).shuffle(files)
        n_val = max(1, int(len(files) * val_frac))
        val_set = set(files[:n_val])
        train_set = set(files[n_val:])

    csv_mtime = os.path.getmtime(csv_path)
    made = 0
    skipped = 0

    print(f"[INFO] Processing {len(groups)} unique filenames from CSV")
    for fname, g in groups:
        # only process files that belong to either train or val split
        if fname not in train_set and fname not in val_set:
            continue

        # locate PDB/CIF
        protein_id = str(fname)

        #find {protein_id}.pdb or {protein_id}.cif in pdb_root
        pdb_file=f"{protein_id}.pdb"
        #add the directory
        pdb_file = os.path.join(pdb_root, pdb_file)
        #check if it exists, if not, try .cif
        if not os.path.isfile(pdb_file):
            pdb_file = os.path.join(pdb_root, f"{protein_id}.cif")
            if not os.path.isfile(pdb_file):
                print(f"[WARN] Could not find PDB/CIF for {protein_id} (tried {protein_id}.pdb and {protein_id}.cif); skipping")
                skipped += 1
                continue
        else:
            print(f"[INFO] Found structure for {protein_id}: {pdb_file}")

        if fname in val_set:
            split = "val"
        elif fname in train_set:
            split = "train"
        else:
            continue

        stem = protein_id
        img_out = os.path.join(images_dir, split, stem + ".png")
        lbl_out = os.path.join(labels_dir, split, stem + ".txt")

        pdb_mtime = os.path.getmtime(pdb_file)

        need_img = force_build or (not os.path.isfile(img_out)) or (os.path.getmtime(img_out) < pdb_mtime)
        need_lbl = force_build or (not os.path.isfile(lbl_out)) or (os.path.getmtime(lbl_out) < csv_mtime)

        # Resume mode: if both artifacts exist, skip preprocessing regardless of mtimes
        if resume and (not force_build) and os.path.isfile(img_out) and os.path.isfile(lbl_out):
            need_img = False
            need_lbl = False

        N = None
        if need_img:
            try:
                D, N = pdb_to_ca_distance_matrix(pdb_file)
            except Exception as e:
                print(f"[WARN] Failed to build image for {pdb_file}: {e}")
                raise e
                skipped += 1
                continue
            save_distance_png_letterboxed(
                    D,
                    img_out,
                    canvas_n=canvas_n,
                    px_per_res=px_per_res,
                    dpi=dpi
                )
        else:
            try:
                N = count_ca_atoms(pdb_file)
            except Exception as e:
                print(f"[WARN] Failed to count CA atoms for {pdb_file}: {e}")
                skipped += 1
                continue

        if need_lbl:
            lines = []
            for _, row in g.iterrows():
                cname = str(row["domain_type"]).strip()
                cid = get_class_id(cname, class_to_id)
                if cid is None:
                    print(f"[INFO] Excluding empty or unknown label '{cname}' in {fname}; skipping row")
                    continue
                if pd.isna(row["domain_start"]) or pd.isna(row["domain_end"]):
                    print(f"[WARN] Missing start/end in {fname}; skipping row")
                    continue
                start_res = int(row["domain_start"])
                end_res = int(row["domain_end"])
                yolo = residue_box_to_yolo(
                    start_res,
                    end_res,
                    protein_n=N,
                    canvas_n=canvas_n
                )
                if yolo is None:
                    print(f"[WARN] Bad box [{start_res},{end_res}] in {fname}; skipping")
                    continue
                cx, cy, w, h = yolo
                lines.append(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            with open(lbl_out, "w") as f:
                f.write("\n".join(lines))

        made += 1

    yaml_path = os.path.join(out_dir, "data.yaml")
    if force_build or (not os.path.isfile(yaml_path)):
        names = class_names
        dataset_cfg = {
            "path": out_dir,
            "train": "images/train",
            "val": "images/val",
            "names": names
        }
        with open(yaml_path, "w") as f:
            yaml.safe_dump(dataset_cfg, f, sort_keys=False)

    print(f"[DONE] Dataset ready at: {out_dir}")
    print(f"  Images/Labels updated for {made} structures; skipped {skipped}.")
    print(f"  data.yaml at: {yaml_path}")

    total = len(groups)
    success_ratio = made / total if total > 0 else 0
    print(f"[SUMMARY] Successful label sets: {made}/{total} ({success_ratio:.1%})")
    return yaml_path

def parse_label_line(line):
    """Parse a YOLO label line into class, box coordinates, and confidence, returning None for malformed lines."""
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    try:
        cls_id = int(float(parts[0]))
        cx = float(parts[1])
        cy = float(parts[2])
        w = float(parts[3])
        h = float(parts[4])
        conf = float(parts[5]) if len(parts) >= 6 else 1.0
        return cls_id, cx, cy, w, h, conf
    except Exception:
        return None
def load_gt_boxes(gt_dir):
    """Load ground-truth YOLO boxes from a labels directory, keyed by filename stem."""
    data = {}
    if not os.path.isdir(gt_dir):
        return data
    for fname in os.listdir(gt_dir):
        if not fname.endswith(".txt"):
            continue
        path = os.path.join(gt_dir, fname)
        boxes = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                rec = parse_label_line(line)
                if rec is None:
                    continue
                cls_id, cx, cy, w, h, _ = rec
                boxes.append((cls_id, cx, cy, w, h))
        stem = os.path.splitext(fname)[0]
        data[stem] = boxes
    return data

def load_pred_boxes(pred_dir):
    """Load predicted YOLO boxes with confidence values from a labels directory, keyed by filename stem."""
    data = {}
    if not os.path.isdir(pred_dir):
        return data
    for fname in os.listdir(pred_dir):
        if not fname.endswith(".txt"):
            continue
        path = os.path.join(pred_dir, fname)
        boxes = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                rec = parse_label_line(line)
                if rec is None:
                    continue
                boxes.append(rec)
        stem = os.path.splitext(fname)[0]
        data[stem] = boxes
    return data


def bbox_iou(gt_box, pred_box):
    """Compute intersection-over-union between one ground-truth box and one predicted box in normalized YOLO format."""
    cls_g, gx, gy, gw, gh = gt_box
    cls_p, px, py, pw, ph, conf = pred_box

    gx1 = gx - gw * 0.5
    gy1 = gy - gh * 0.5
    gx2 = gx + gw * 0.5
    gy2 = gy + gh * 0.5

    px1 = px - pw * 0.5
    py1 = py - ph * 0.5
    px2 = px + pw * 0.5
    py2 = py + ph * 0.5

    ix1 = max(gx1, px1)
    iy1 = max(gy1, py1)
    ix2 = min(gx2, px2)
    iy2 = min(gy2, py2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0

    area_g = gw * gh
    area_p = pw * ph
    union = area_g + area_p - inter
    if union <= 0.0:
        return 0.0
    return inter / union

def evaluate_predictions(
    gt_dir,
    pred_dir,
    conf_thr=0.25,
    iou_thr=0.5,
    return_details=False
):
    """Evaluate predicted YOLO boxes against ground truth using confidence and IoU thresholds."""
    gt_dict = load_gt_boxes(gt_dir)
    pred_dict_all = load_pred_boxes(pred_dir)

    tp = 0
    fp = 0
    fn = 0

    confusion = defaultdict(lambda: defaultdict(int))
    missed_gt = Counter()
    fp_per_class = Counter()
    classes_seen = set()

    nae_sum = 0.0
    nae_count = 0

    for stem, gts in gt_dict.items():
        preds_all = pred_dict_all.get(stem, [])
        preds = [p for p in preds_all if p[5] >= conf_thr]

        for gt in gts:
            classes_seen.add(gt[0])
        for p in preds:
            classes_seen.add(p[0])

        gt_counts = Counter(b[0] for b in gts)
        pred_counts = Counter(p[0] for p in preds)
        gt_total = sum(gt_counts.values())
        if gt_total > 0:
            allc = set(gt_counts.keys()) | set(pred_counts.keys())
            abs_err = 0
            for c in allc:
                abs_err += abs(pred_counts.get(c, 0) - gt_counts.get(c, 0))
            nae_sum += abs_err / float(gt_total)
            nae_count += 1

        used_pred = [False] * len(preds)

        # GT-driven matching, IoU-based
        for gt in gts:
            gt_cls = gt[0]
            best_iou = 0.0
            best_j = -1
            for j, p in enumerate(preds):
                if used_pred[j]:
                    continue
                iou_val = bbox_iou(gt, p)
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_j = j
            if best_j >= 0 and best_iou >= iou_thr:
                pred_cls = preds[best_j][0]
                confusion[gt_cls][pred_cls] += 1
                used_pred[best_j] = True
            else:
                missed_gt[gt_cls] += 1

        for j, used in enumerate(used_pred):
            if not used:
                p_cls = preds[j][0]
                fp_per_class[p_cls] += 1

    classes = sorted(classes_seen)

    for c in classes:
        tp += confusion[c].get(c, 0)

    misclassified = 0
    for gt_c in classes:
        for pred_c in classes:
            if pred_c == gt_c:
                continue
            misclassified += confusion[gt_c].get(pred_c, 0)

    fn = sum(missed_gt.values()) + misclassified
    fp = sum(fp_per_class.values()) + misclassified

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall > 0 else 0.0
    denom = tp + fp + fn
    accuracy = tp / denom if denom > 0 else 0.0
    nae = nae_sum / nae_count if nae_count > 0 else float("nan")

    print("[EVAL] conf_thr =", conf_thr, "iou_thr =", iou_thr)
    print("[EVAL] TP =", tp, "FP =", fp, "FN =", fn)
    print("[EVAL] Precision = {:.4f}".format(precision))
    print("[EVAL] Recall    = {:.4f}".format(recall))
    print("[EVAL] F1        = {:.4f}".format(f1))
    print("[EVAL] Accuracy  = {:.4f}".format(accuracy))
    print("[EVAL] NAE       = {:.4f}".format(nae))

    result = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "nae": nae,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }

    if return_details:
        result["confusion"] = confusion
        result["missed_gt"] = missed_gt
        result["fp_per_class"] = fp_per_class
        result["classes"] = classes

    return result
def print_contingency_tables(confusion, missed_gt, fp_per_class, classes):
    """Print class-level contingency tables for matched, missed, and spurious predictions."""
    print()
    print("[CONTINGENCY] GT (rows) vs Pred (cols)")
    header = ["GT\\Pred"] + [str(c) for c in classes] + ["missed"]
    print("\t".join(header))
    for gt_c in classes:
        row = [str(gt_c)]
        for pred_c in classes:
            row.append(str(confusion[gt_c].get(pred_c, 0)))
        row.append(str(missed_gt.get(gt_c, 0)))
        print("\t".join(row))

    print()
    print("[CONTINGENCY] Spurious predictions (no GT) per predicted class")
    row = ["spurious"]
    for pred_c in classes:
        row.append(str(fp_per_class.get(pred_c, 0)))
    print("\t".join(row))
def run_val_prediction_and_evaluate(args):
    """Run YOLO prediction on validation images with the selected weights and evaluate the resulting labels."""
    import subprocess

    run_dir = os.path.join(args.project, args.name)
    weights_dir = os.path.join(run_dir, 'weights')

    best_path = os.path.join(weights_dir, 'best.pt')
    last_path = os.path.join(weights_dir, 'last.pt')

    if os.path.isfile(best_path):
        model_path = best_path
    elif os.path.isfile(last_path):
        model_path = last_path
    else:
        print('[WARN] No best.pt or last.pt found; falling back to args.weights')
        model_path = args.weights

    val_dir = os.path.join(args.out, 'images', 'val')
    if not os.path.isdir(val_dir):
        print('[WARN] Validation image directory not found:', val_dir)
        return

    pred_project = os.path.join(run_dir, 'predictions')
    pred_name = 'val'
    os.makedirs(pred_project, exist_ok=True)

    cmd = (
        'yolo detect predict '
        'model={} '
        'source={} '
        'conf={} '
        'save_txt=True '
        'save_conf=True '
        'project={} '
        'name={}'
    ).format(model_path, val_dir, args.confidence, pred_project, pred_name)

    print('[RUN]', cmd)
    subprocess.run(cmd, shell=True, check=True)

    gt_dir = os.path.join(args.out, 'labels', 'val')
    pred_labels_dir = os.path.join(pred_project, pred_name, 'labels')

    evaluate_predictions(
        gt_dir=gt_dir,
        pred_dir=pred_labels_dir,
        conf_thr=args.confidence,
        iou_thr=0.5
    )

def main():
    """Parse command-line arguments and run dataset preparation, training, prediction, and evaluation workflows."""
    ap = argparse.ArgumentParser()

    # Data / IO
    ap.add_argument(
        '--csv',
        default='test_table.csv',
        help='Annotations TSV (tab-separated).'
    )
    ap.add_argument(
        '--pdb_root',
        default='test/data',
        help='Root folder containing PDB/CIF files referenced in CSV filename.'
    )
    ap.add_argument(
        '--out',
        default='/oceanstor/scratch/tllseedorf/e1103389/adhesin/train_results',
        help='Output dataset folder.'
    )

    # Dataset / splitting
    ap.add_argument('--px_per_res', type=int, default=5)
    ap.add_argument('--dpi', type=int, default=100)
    ap.add_argument('--val_frac', type=float, default=0.15)
    ap.add_argument('--seed', type=int, default=1337)
    ap.add_argument(
        '--force_build',
        action='store_true',
        help='Force rebuild images/labels/data.yaml even if fresh.'
    )

    # YOLO training args
    ap.add_argument('--weights', default='yolo26n.pt')
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--project', default='adhesin_yolo')
    ap.add_argument('--target_domain', default=None, help='Deprecated/ignored. This multiclass script uses all domain types found in the annotation file.')
    ap.add_argument('--name', default='train_from_annotations')
    ap.add_argument(
        '--run',
        default=None,
        help='YOLO run subdirectory under --project. If omitted or missing, the latest modified run with best.pt/last.pt is reused when available.'
    )

    # Resume behaviour
    ap.add_argument(
        '--resume',
        dest='resume',
        action='store_true',
        help='Resume training and data preparation where possible. (default)'
    )
    ap.add_argument(
        '--no-resume',
        dest='resume',
        action='store_false',
        help='Start from scratch, deleting prior data and training runs.'
    )
    ap.set_defaults(resume=True)

    # Fast-path flags (reuse prior artifacts)
    ap.add_argument(
        '--skip_build',
        action='store_true',
        help='Skip dataset building; use existing data.yaml under --out.'
    )
    ap.add_argument(
        '--skip_train',
        action='store_true',
        help='Skip YOLO training; reuse existing run weights (best.pt/last.pt) or --model.'
    )
    ap.add_argument(
        '--skip_predict',
        action='store_true',
        help='Skip prediction; reuse existing predicted label .txt files for evaluation.'
    )
    ap.add_argument(
        '--eval_only',
        action='store_true',
        help='Only evaluate existing predictions vs GT; implies --skip_build --skip_train --skip_predict.'
    )
    ap.add_argument(
        '--model',
        default=None,
        help='Explicit model weights path (.pt) to use for prediction (overrides best.pt/last.pt discovery).'
    )
    ap.add_argument(
        '--pred_labels_dir',
        default=None,
        help='Explicit directory containing predicted label .txt files (YOLO save_txt output). If set, used for evaluation.'
    )

    # Evaluation / CV args
    ap.add_argument(
        '--conf_list',
        type=float,
        nargs='+',
        default=[0.1,0.2,0.4,  0.5, 0.7,0.8,1],
        help='List of confidence thresholds to evaluate.'
    )
    ap.add_argument(
        '--k_folds',
        type=int,
        default=1,
        help='Number of folds for cross-validation. If 1, run a single train/eval split.'
    )
    ap.add_argument(
        '--iou_thr',
        type=float,
        default=0.1,
        help='IoU threshold for evaluation.'
    )

    args = ap.parse_args()

    # Resolve project/out to absolute paths; if out is relative, treat it as relative to project dir.
    args.project = os.path.abspath(args.project)
    if not os.path.isabs(args.out):
        args.out = os.path.abspath(os.path.join(args.project, args.out))
    else:
        args.out = os.path.abspath(args.out)
    # Also normalize csv/pdb_root for consistency
    args.csv = os.path.abspath(args.csv)
    args.pdb_root = os.path.abspath(args.pdb_root)

    # Normalize project/out paths early so subprocess(yolo) sees absolute dataset paths
    args.project = os.path.abspath(args.project)
    args.out = os.path.abspath(args.out)

    requested_run = args.run
    latest_run = find_latest_run_with_weights(args.project)

    if requested_run is None:
        if latest_run is not None:
            args.run = latest_run
            args.name = latest_run
            print(f'[INFO] --run not specified; reusing latest weighted run: {args.run}')
        else:
            args.run = args.name
            print(f'[INFO] --run not specified and no weighted run found; using --name: {args.name}')
    else:
        requested_run_dir = os.path.join(args.project, requested_run)
        if os.path.isdir(requested_run_dir):
            args.name = requested_run
            print(f'[INFO] Using requested run: {args.name}')
        elif latest_run is not None:
            args.run = latest_run
            args.name = latest_run
            print(
                f'[INFO] Requested run not found: {requested_run_dir}; '
                f'reusing latest weighted run instead: {args.name}'
            )
        else:
            args.name = requested_run
            print(
                f'[INFO] Requested run not found and no weighted run found; '
                f'using requested run name for a new run: {args.name}'
            )

    def _die(msg, code=2):
        """Print an error message and terminate execution with the requested exit code."""
        print(f"[ERROR] {msg}")
        raise SystemExit(code)

    if args.eval_only:
        args.skip_build = True
        args.skip_train = True
        args.skip_predict = True
        args.resume = True

    # If k_folds > 1, run k-fold CV (build/train/predict/eval per fold) and exit.
    if args.k_folds > 1:
        run_kfold_cv(args)
        return

    # Single train/val split path
    run_dir = os.path.join(args.project, args.name)

    if not args.resume:
        print('[INFO] --no-resume specified. Deleting output directories to start from scratch.')
        if os.path.isdir(args.out):
            print('[INFO] Deleting dataset output directory:', args.out)
            shutil.rmtree(args.out)
        if os.path.isdir(run_dir):
            print('[INFO] Deleting YOLO run directory:', run_dir)
            shutil.rmtree(run_dir)

    os.makedirs(args.out, exist_ok=True)

    # Build (or reuse) dataset
    if args.skip_build:
        yaml_path = os.path.join(args.out, "data.yaml")
        if not os.path.isfile(yaml_path):
            _die(f"--skip_build was set but data.yaml was not found at: {yaml_path}")
        print("[INFO] --skip_build: Reusing existing dataset config:", yaml_path)
    else:
        yaml_path = build_dataset(
            csv_path=args.csv,
            pdb_root=args.pdb_root,
            out_dir=args.out,
            px_per_res=args.px_per_res,
            dpi=args.dpi,
            val_frac=args.val_frac,
            seed=args.seed,
            force_build=args.force_build,
            resume=args.resume
        )

    # Train YOLO once on this split
    import subprocess

        # Train YOLO once on this split
    import subprocess

    cmd_train = (
        'yolo detect train '
        f'model={args.weights} '
        f'data=\"{yaml_path}\" '
        f'imgsz={args.imgsz} '
        f'epochs={args.epochs} '
        f'batch={args.batch} '
        f'project=\"{args.project}\" '
        f'name=\"{args.name}\" '
        'exist_ok=True'
    )

    auto_skip_train = (
        args.resume
        and not args.force_build
        and os.path.isdir(run_dir)
        and find_model_in_run(run_dir) is not None
    )

    if auto_skip_train:
        print(
            '[INFO] Existing project directory with .pt weights detected; '
            'automatically skipping training.'
        )
        args.skip_train = True

    print('[RUN]', cmd_train)

    if args.skip_train:
        print('[INFO] --skip_train: Skipping training; will reuse existing weights if available.')

        if args.model is None or not os.path.isfile(args.model):
            existing_model = find_model_in_run(run_dir)
            if existing_model is None:
                weights_dir_check = os.path.join(args.project, args.name, 'weights')
                best_check = os.path.join(weights_dir_check, 'best.pt')
                last_check = os.path.join(weights_dir_check, 'last.pt')
                _die(
                    '--skip_train was set but no prior weights were found under the project run directory. '
                    f'Tried: {best_check} and {last_check}. '
                    'Either run training once (without --skip_train) or provide --model /path/to/weights.pt.'
                )
            args.model = existing_model
            print('[INFO] Using existing run weights:', args.model)
    else:
        try:
            subprocess.run(cmd_train, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            raise e

    # Choose weights for prediction/evaluation. Prefer explicit --model only if it exists.
    if args.model is not None and os.path.isfile(args.model):
        model_path = args.model
        print("[INFO] Using --model:", model_path)
    else:
        if args.model is not None:
            print(f'[WARN] --model was provided but does not exist: {args.model}')
        model_path = find_model_in_run(run_dir)
        if model_path is not None:
            print('[INFO] Using discovered run weights:', model_path)
        else:
            print('[WARN] No best.pt or last.pt found in project/run; falling back to args.weights')
            model_path = args.weights

    # Predict on validation images with conf=0.0 (collect all boxes), unless skipped
    val_img_dir = os.path.join(args.out, 'images', 'val')
    pred_project = os.path.join(args.project, args.name, 'predictions_all')
    pred_name = 'val_all'

    if args.skip_predict:
        print('[INFO] --skip_predict: Skipping prediction; will reuse existing prediction labels if available.')
        # Enforce that predicted labels exist (under the project run directory unless --pred_labels_dir is provided).
        if args.pred_labels_dir is not None:
            pred_labels_dir_check = args.pred_labels_dir
        else:
            pred_labels_dir_check = os.path.join(pred_project, pred_name, 'labels')
        if not os.path.isdir(pred_labels_dir_check):
            _die(
                '--skip_predict was set but predicted labels directory was not found. '
                f'Tried: {pred_labels_dir_check}. '
                'Either run prediction once (without --skip_predict) or point to an existing folder via --pred_labels_dir.'
            )
        # Basic sanity: must contain at least one .txt
        has_txt = any(fn.endswith('.txt') for fn in os.listdir(pred_labels_dir_check))
        if not has_txt:
            _die(
                '--skip_predict was set but no .txt prediction files were found in: '
                f'{pred_labels_dir_check}'
            )
    else:
        if not os.path.isdir(val_img_dir):
            print('[WARN] Validation image directory not found:', val_img_dir)
            return

        os.makedirs(pred_project, exist_ok=True)

        cmd_pred = (
            'yolo detect predict '
            f'model=\"{model_path}\" '
            f'source=\"{val_img_dir}\" '
            'conf=0.0 '
            'save_txt=True '
            'save_conf=True '
            f'project=\"{pred_project}\" '
            f'name=\"{pred_name}\" '
            'exist_ok=True'
        )
        print('[RUN]', cmd_pred)
        subprocess.run(cmd_pred, shell=True, check=True)

    gt_dir = os.path.join(args.out, 'labels', 'val')
    if not os.path.isdir(gt_dir):
        _die(f"Ground-truth labels directory not found: {gt_dir}")
    if args.pred_labels_dir is not None:
        pred_labels_dir = args.pred_labels_dir
        print('[INFO] Using --pred_labels_dir for evaluation:', pred_labels_dir)
    else:
        pred_labels_dir = os.path.join(pred_project, pred_name, 'labels')

    # Evaluate at each confidence, record metrics, and pick best by F1 score
    best_conf = None
    best_f1 = None
    single_results = []

    print()
    print('[EVAL] Single split metrics for each confidence:')
    for conf_thr in args.conf_list:
        metrics = evaluate_predictions(
            gt_dir=gt_dir,
            pred_dir=pred_labels_dir,
            conf_thr=conf_thr,
            iou_thr=args.iou_thr,
            return_details=False
        )
        single_results.append((conf_thr, metrics))
        f1_val = metrics['f1']
        if best_f1 is None or f1_val > best_f1:
            best_f1 = f1_val
            best_conf = conf_thr

    # Print a simple table of conf vs metrics
    print()
    print('conf\tprecision\trecall\tF1\taccuracy\tNAE')
    for conf_thr, m in single_results:
        nae_val = m['nae']
        nae_txt = f'{nae_val:.4f}' if not math.isnan(nae_val) else 'NaN'
        print(
            f'{conf_thr:.3f}\t'
            f'{m["precision"]:.4f}\t'
            f'{m["recall"]:.4f}\t'
            f'{m["f1"]:.4f}\t'
            f'{m["accuracy"]:.4f}\t'
            f'{nae_txt}'
        )

    if best_conf is None:
        print()
        print('[EVAL] Could not determine best confidence.')
        return

    print()
    print(f'[EVAL] Best confidence on this split by F1: conf = {best_conf:.3f}, F1 = {best_f1:.6f}')

    # Final detailed eval at best_conf: contingency tables and GT vs Pred
    final_metrics = evaluate_predictions(
        gt_dir=gt_dir,
        pred_dir=pred_labels_dir,
        conf_thr=best_conf,
        iou_thr=args.iou_thr,
        return_details=True
    )
    confusion = final_metrics['confusion']
    missed_gt = final_metrics['missed_gt']
    fp_per_class = final_metrics['fp_per_class']
    classes = final_metrics['classes']

    print_contingency_tables(confusion, missed_gt, fp_per_class, classes)




def run_val_prediction(yaml_path, args):
    """Run YOLO prediction on the validation image folder and save predictions under the selected run directory."""
    import subprocess

    # Use best.pt if available, else last.pt, else fallback to initial weights
    run_dir = os.path.join(args.project, args.name)
    weights_dir = os.path.join(run_dir, "weights")

    best_path = os.path.join(weights_dir, "best.pt")
    last_path = os.path.join(weights_dir, "last.pt")

    if os.path.isfile(best_path):
        model_path = best_path
    elif os.path.isfile(last_path):
        model_path = last_path
    else:
        print("[WARN] No best.pt or last.pt found; falling back to args.weights")
        model_path = args.weights

    # Validation images directory from the dataset we just built
    val_dir = os.path.join(args.out, "images", "val")
    if not os.path.isdir(val_dir):
        print(f"[WARN] Validation image directory not found: {val_dir}")
        return

    # Save predictions under the run folder
    pred_project = os.path.join(run_dir, "predictions")
    pred_name = "val"
    os.makedirs(pred_project, exist_ok=True)

    cmd = (
        f"yolo detect predict "
        f"model={model_path} "
        f"source={val_dir} "
        f"conf={args.confidence} "
        f"project={pred_project} "
        f"name={pred_name} "
        f"exist_ok=True"
    )
    print("[RUN]", cmd)
    subprocess.run(cmd, shell=True, check=True)

if __name__ == "__main__":
    main()


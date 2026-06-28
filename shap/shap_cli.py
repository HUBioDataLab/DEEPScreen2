#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SHAP Visualization CLI
======================

When you provide a single model folder (e.g. shap_numpy_data/CHEMBL301_cnn) it
generates the processed PNGs for every molecule in that folder and packs them
into a ZIP.

When you provide a parent folder (e.g. shap_numpy_data) it processes each model
subfolder underneath it separately and produces a separate ZIP for each one
(e.g. CHEMBL301_cnn.zip, CHEMBL301_yolo.zip).
"""

import argparse
import os
import re
import sys
import zipfile
from glob import glob
from io import BytesIO

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

cmap_rw = LinearSegmentedColormap.from_list("white_red", ["white", "red"])


def apply_attention_style(heatmap, sigma=3):
    if sigma > 0:
        heatmap = cv2.GaussianBlur(heatmap, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return heatmap


def robust_normalize(data, p_low=1, p_high=99):
    vmin = np.percentile(data, p_low)
    vmax = np.percentile(data, p_high)
    if vmax - vmin < 1e-9:
        return np.zeros_like(data)
    data = np.clip(data, vmin, vmax)
    return (data - vmin) / (vmax - vmin)


def rotate_image_back(image, angle_degrees, is_shap=False):
    if angle_degrees == 0:
        return image
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, -angle_degrees, 1.0)
    border_val = 0 if is_shap else (1.0, 1.0, 1.0)
    return cv2.warpAffine(
        image, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_val,
    )


def load_and_aggregate_data(base_name, input_dir):
    all_files = glob(os.path.join(input_dir, "*.npz"))
    pattern = re.compile(re.escape(base_name) + r"_(\d+)\.npz$")
    matched_files = []
    for f in all_files:
        match = pattern.search(os.path.basename(f))
        if match:
            matched_files.append((f, int(match.group(1))))
    if not matched_files:
        return None

    shap_list, img_list, probs_list = [], [], []
    for f_path, angle in matched_files:
        data = np.load(f_path)
        raw_shap = data["shap_values"].transpose((1, 2, 0))
        raw_img = data["image"].transpose((1, 2, 0))
        shap_list.append(rotate_image_back(raw_shap, angle, is_shap=True))
        img_list.append(rotate_image_back(raw_img, angle, is_shap=False))
        probs_list.append(data["probabilities"])

    return {
        "shap": np.mean(shap_list, axis=0),
        "image": np.mean(img_list, axis=0),
        "probs": np.mean(probs_list, axis=0),
    }


def load_smiles_lookup(molecules_csv_path):
    if not molecules_csv_path or not os.path.exists(molecules_csv_path):
        return {}
    import csv
    lookup = {}
    with open(molecules_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mol_id = str(row.get("molecule_id", "")).strip()
            smiles = str(row.get("smiles", "")).strip()
            if mol_id and smiles and smiles.lower() != "nan":
                lookup[mol_id] = smiles
    return lookup


def generate_missing_transparent_png(base_name, mol_images_dir, smiles_lookup, target_size=300):
    smiles = smiles_lookup.get(base_name)
    if not smiles:
        return None
    try:
        from generate_images import smiles_to_transparent_png
    except Exception:
        return None
    os.makedirs(mol_images_dir, exist_ok=True)
    out_path = os.path.join(mol_images_dir, f"{base_name}.png")
    try:
        img = smiles_to_transparent_png(smiles, size=target_size)
        img.save(out_path, format="PNG")
        return out_path
    except Exception:
        return None


def load_transparent_mol_image(mol_images_dir, base_name, smiles_lookup, target_size=300):
    path = os.path.join(mol_images_dir, f"{base_name}.png")
    if not os.path.exists(path):
        generated = generate_missing_transparent_png(base_name, mol_images_dir, smiles_lookup, target_size)
        if generated is None:
            return None
        path = generated
    img = Image.open(path).convert("RGBA")
    if img.size != (target_size, target_size):
        img = img.resize((target_size, target_size), Image.LANCZOS)
    return np.array(img).astype(np.float32) / 255.0


def compute_shap_norm(data, blur_sigma, gamma, hotspot_p):
    shap_sum = np.abs(data["shap"]).sum(axis=2)
    shap_smooth = apply_attention_style(shap_sum, sigma=blur_sigma)
    thresh = np.percentile(shap_smooth, hotspot_p)
    shap_clipped = np.where(shap_smooth > thresh, shap_smooth, thresh)
    shap_norm = robust_normalize(shap_clipped)
    shap_norm = np.power(shap_norm, gamma)
    return shap_norm


def create_download_plot(data, blur_sigma, gamma, hotspot_p, alpha, mol_rgba=None):
    """Identical to the Download ZIP flow in app7.py."""
    if mol_rgba is not None:
        mol_rgb = mol_rgba[:, :, :3]
    else:
        mol_rgb = np.clip(data["image"], 0, 1)

    shap_norm = compute_shap_norm(data, blur_sigma, gamma, hotspot_p)
    heatmap_rgb = cmap_rw(shap_norm)[..., :3]
    final_combined = heatmap_rgb * mol_rgb

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(final_combined)
    ax.axis("off")
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    return fig


def collect_base_names(input_dir):
    all_npz = glob(os.path.join(input_dir, "*.npz"))
    if not all_npz:
        return []
    return sorted({re.sub(r"_\d+\.npz$", "", os.path.basename(f)) for f in all_npz})


def process_directory(input_dir, output_zip, mol_images_dir, molecules_csv_path,
                      blur_sigma, gamma, hotspot_p, alpha):
    base_names = collect_base_names(input_dir)
    if not base_names:
        print(f"  ! No NPZ files found: {input_dir}")
        return False

    smiles_lookup = load_smiles_lookup(molecules_csv_path)
    os.makedirs(os.path.dirname(os.path.abspath(output_zip)) or ".", exist_ok=True)

    print(f"  -> {len(base_names)} molecules to process -> {output_zip}")
    written = 0
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, name in enumerate(base_names, 1):
            data = load_and_aggregate_data(name, input_dir)
            if data is None:
                print(f"    [{i}/{len(base_names)}] {name}: no data, skipped")
                continue
            mol_rgba = load_transparent_mol_image(mol_images_dir, name, smiles_lookup)
            fig = create_download_plot(data, blur_sigma, gamma, hotspot_p, alpha, mol_rgba)
            img_buf = BytesIO()
            fig.savefig(img_buf, format="png", dpi=300, bbox_inches="tight",
                        pad_inches=0, facecolor="white")
            plt.close(fig)
            img_buf.seek(0)
            zf.writestr(f"{name}_attention.png", img_buf.getvalue())
            written += 1
            print(f"    [{i}/{len(base_names)}] {name}: ok")
    print(f"  ✓ {written} images written -> {output_zip}")
    return written > 0


def find_model_subdirs(parent):
    """Finds subfolders under parent that contain .npz files."""
    subs = []
    for entry in sorted(os.listdir(parent)):
        full = os.path.join(parent, entry)
        if os.path.isdir(full) and glob(os.path.join(full, "*.npz")):
            subs.append(full)
    return subs


def main():
    p = argparse.ArgumentParser(
        description="Terminal version of the app7.py Download-ZIP flow",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", "-i", required=True,
                   help="NPZ folder, or a parent folder containing model subfolders")
    p.add_argument("--output_dir", "-o", default="./output_zips",
                   help="Folder where the ZIPs will be written")
    p.add_argument("--mol_images_dir", default="./mol_images",
                   help="Transparent molecule PNGs")
    p.add_argument("--molecules_csv", default="./molecules.csv",
                   help="SMILES source for missing PNGs")
    p.add_argument("--hotspot_p", type=float, default=95.0,
                   help="Focus threshold (Top %%) — Streamlit slider")
    p.add_argument("--blur_sigma", type=float, default=5.7,
                   help="Smoothing (Blur sigma)")
    p.add_argument("--gamma", type=float, default=1.50,
                   help="Intensity (Gamma)")
    p.add_argument("--alpha", type=float, default=0.75,
                   help="Opacity")
    args = p.parse_args()

    if not os.path.isdir(args.input):
        print(f"ERROR: input folder does not exist: {args.input}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    direct_npz = bool(glob(os.path.join(args.input, "*.npz")))
    targets = []
    if direct_npz:
        targets.append(args.input)
    else:
        targets = find_model_subdirs(args.input)
        if not targets:
            print(f"ERROR: no .npz found under {args.input}", file=sys.stderr)
            sys.exit(1)

    print(f"Parameters: hotspot_p={args.hotspot_p} blur={args.blur_sigma} "
          f"gamma={args.gamma} alpha={args.alpha}")
    print(f"Number of folders to process: {len(targets)}")

    any_ok = False
    for sub in targets:
        label = os.path.basename(os.path.normpath(sub))
        out_zip = os.path.join(args.output_dir, f"{label}.zip")
        print(f"\n== {label} ==")
        ok = process_directory(
            input_dir=sub,
            output_zip=out_zip,
            mol_images_dir=args.mol_images_dir,
            molecules_csv_path=args.molecules_csv,
            blur_sigma=args.blur_sigma,
            gamma=args.gamma,
            hotspot_p=args.hotspot_p,
            alpha=args.alpha,
        )
        any_ok = any_ok or ok

    sys.exit(0 if any_ok else 2)


if __name__ == "__main__":
    main()

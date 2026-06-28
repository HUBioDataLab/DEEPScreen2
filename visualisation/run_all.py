#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Batch driver: extract embeddings for every trained model under a directory, then
(optionally) plot per-model grids. Task id and architecture are auto-parsed from
each filename (canonical: TASK_best_val-TASK-<MODELNAME>-<fc1>-<fc2>-...).

Example
-------
python visualisation/run_all.py \
    --models_dir trained_models \
    --split test \
    --emb_dir visualisation/embeddings \
    --fig_dir visualisation/figures \
    --methods tsne umap \
    --cuda 0
"""
import os
import re
import sys
import glob
import argparse
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_task_and_model(path):
    name = os.path.basename(path)
    model_name = next(
        (k for k in ("CNNModel1", "CNNModel2", "ViT", "YOLOv11") if k in name), None
    )
    # Canonical: "<TASK>_best_val-<TASK>-<MODEL>-..."
    m = re.match(r"(.+?)_best_val-([^-]+)-", name)
    task = m.group(2) if m else name.split("-")[0]
    return task, model_name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models_dir", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--emb_dir", default=os.path.join(HERE, "embeddings"))
    ap.add_argument("--fig_dir", default=os.path.join(HERE, "figures"))
    ap.add_argument("--methods", nargs="+", default=["tsne", "umap"])
    ap.add_argument("--cuda", type=int, default=0)
    ap.add_argument("--skip_plots", action="store_true")
    args = ap.parse_args()

    pths = sorted(glob.glob(os.path.join(args.models_dir, "**", "*.pth"), recursive=True))
    if not pths:
        raise SystemExit(f"No .pth under {args.models_dir}")

    print(f"Found {len(pths)} model files.")
    for p in pths:
        task, model_name = parse_task_and_model(p)
        if model_name is None:
            print(f"[skip] cannot infer architecture: {os.path.basename(p)}")
            continue
        print(f"\n>>> {task} / {model_name}")
        cmd = [
            sys.executable, os.path.join(HERE, "extract_embeddings.py"),
            "--model_path", p, "--task", task, "--split", args.split,
            "--out_dir", args.emb_dir, "--cuda", str(args.cuda),
            "--model_name", model_name,
        ]
        rc = subprocess.call(cmd)
        if rc != 0:
            print(f"[warn] extraction failed (rc={rc}) for {p}")

    if args.skip_plots:
        return

    # Per-architecture grids
    for model_name in ("CNNModel1", "CNNModel2", "ViT", "YOLOv11"):
        for method in args.methods:
            cmd = [
                sys.executable, os.path.join(HERE, "plot_projections.py"), "grid",
                "--inputs", args.emb_dir, "--model_name", model_name,
                "--method", method, "--out_dir", args.fig_dir,
            ]
            subprocess.call(cmd)
    # Individual panels for everything
    cmd = [
        sys.executable, os.path.join(HERE, "plot_projections.py"), "single",
        "--inputs", args.emb_dir, "--methods", *args.methods,
        "--out_dir", args.fig_dir,
    ]
    subprocess.call(cmd)


if __name__ == "__main__":
    main()

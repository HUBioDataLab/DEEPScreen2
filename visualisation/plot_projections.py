#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Projection Plotter
==================
Loads embedding .npz files written by extract_embeddings.py, projects them to 2D
with t-SNE and/or UMAP, and renders scatter plots colored by activity
(red = active, blue = inactive) in the style of the reference figure.

Two layouts:
  * single  : one figure per (task, model) .npz
  * grid    : a multi-panel grid across tasks for a given model + method
              (rows auto-flow), mirroring the reference multi-task figure.

Examples
--------
# All npz in a folder, both methods, individual panels:
python visualisation/plot_projections.py single \
    --inputs visualisation/embeddings \
    --methods tsne umap \
    --out_dir visualisation/figures

# Grid across all CNN tasks, t-SNE:
python visualisation/plot_projections.py grid \
    --inputs visualisation/embeddings \
    --model_name CNNModel1 --method tsne \
    --ncols 4 --out_dir visualisation/figures
"""

import os
import sys
import glob
import argparse
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")

ACTIVE_COLOR = "#d62728"    # red
INACTIVE_COLOR = "#1f77b4"  # blue


# -----------------------------------------------------------------------------
# Loading
# -----------------------------------------------------------------------------
def collect_npz(inputs):
    paths = []
    for inp in inputs:
        if os.path.isdir(inp):
            paths.extend(sorted(glob.glob(os.path.join(inp, "*.npz"))))
        elif inp.endswith(".npz"):
            paths.append(inp)
    return paths


def load_npz(path):
    d = np.load(path, allow_pickle=True)
    return dict(
        embeddings=d["embeddings"],
        labels=d["labels"].astype(int),
        task=str(d["task"]),
        model_name=str(d["model_name"]),
        split=str(d["split"]),
    )


# -----------------------------------------------------------------------------
# Projection
# -----------------------------------------------------------------------------
def project(embeddings, method, seed=42):
    n = embeddings.shape[0]
    if n < 3:
        raise ValueError(f"Too few points to project: {n}")

    # Standardize features for stable projection
    X = embeddings.astype(np.float64)
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)

    if method == "tsne":
        from sklearn.manifold import TSNE
        perplexity = max(5, min(30, (n - 1) // 3))
        return TSNE(
            n_components=2, perplexity=perplexity, init="pca",
            learning_rate="auto", random_state=seed,
        ).fit_transform(X)

    if method == "umap":
        try:
            import umap
        except ImportError as e:
            raise SystemExit(
                "umap-learn not installed. `pip install umap-learn` or use --methods tsne."
            ) from e
        n_neighbors = max(5, min(15, n - 1))
        return umap.UMAP(
            n_components=2, n_neighbors=n_neighbors, min_dist=0.1,
            random_state=seed,
        ).fit_transform(X)

    raise ValueError(f"Unknown method: {method}")


# -----------------------------------------------------------------------------
# Drawing
# -----------------------------------------------------------------------------
def draw_scatter(ax, coords, labels, title=None, point_size=8):
    act = labels == 1
    inact = labels == 0
    ax.scatter(coords[inact, 0], coords[inact, 1], s=point_size,
               c=INACTIVE_COLOR, alpha=0.55, linewidths=0, label="Inactive")
    ax.scatter(coords[act, 0], coords[act, 1], s=point_size,
               c=ACTIVE_COLOR, alpha=0.65, linewidths=0, label="Active")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    if title:
        ax.set_title(title, fontsize=11)


def legend_handles():
    return [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=ACTIVE_COLOR,
               markersize=8, label="Active"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=INACTIVE_COLOR,
               markersize=8, label="Inactive"),
    ]


# -----------------------------------------------------------------------------
# Modes
# -----------------------------------------------------------------------------
def run_single(args):
    paths = collect_npz(args.inputs)
    if not paths:
        raise SystemExit("No .npz inputs found.")
    os.makedirs(args.out_dir, exist_ok=True)

    for p in paths:
        data = load_npz(p)
        for method in args.methods:
            try:
                coords = project(data["embeddings"], method, args.seed)
            except Exception as e:
                print(f"[skip] {os.path.basename(p)} ({method}): {e}")
                continue
            fig, ax = plt.subplots(figsize=(5, 5))
            title = f"{data['task']} · {data['model_name']} · {method.upper()}"
            draw_scatter(ax, coords, data["labels"], title=title)
            ax.legend(handles=legend_handles(), loc="best", frameon=False, fontsize=9)
            fig.tight_layout()
            out = os.path.join(
                args.out_dir,
                f"{data['task']}__{data['model_name']}__{data['split']}__{method}.png",
            )
            fig.savefig(out, dpi=200, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved -> {out}")


def run_grid(args):
    paths = collect_npz(args.inputs)
    # filter by model_name if requested
    items = []
    for p in paths:
        d = load_npz(p)
        if args.model_name and d["model_name"] != args.model_name:
            continue
        items.append(d)
    if not items:
        raise SystemExit("No matching .npz for grid.")

    items.sort(key=lambda d: d["task"])
    n = len(items)
    ncols = args.ncols
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.2 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, d in zip(axes, items):
        try:
            coords = project(d["embeddings"], args.method, args.seed)
            draw_scatter(ax, coords, d["labels"], title=d["task"], point_size=6)
        except Exception as e:
            ax.set_title(f"{d['task']}\n(skip: {e})", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
    for ax in axes[n:]:
        ax.axis("off")

    fig.legend(handles=legend_handles(), loc="lower right", frameon=False, fontsize=11)
    model_tag = args.model_name or "all"
    fig.suptitle(f"{args.method.upper()} embeddings — {model_tag}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"grid__{model_tag}__{args.method}.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out}")


def main():
    ap = argparse.ArgumentParser(description="Plot t-SNE/UMAP projections.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--inputs", nargs="+", required=True,
                        help="Folders and/or .npz files.")
    common.add_argument("--out_dir", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "figures"))
    common.add_argument("--seed", type=int, default=42)

    s = sub.add_parser("single", parents=[common])
    s.add_argument("--methods", nargs="+", default=["tsne", "umap"],
                   choices=["tsne", "umap"])

    g = sub.add_parser("grid", parents=[common])
    g.add_argument("--model_name", default=None)
    g.add_argument("--method", default="tsne", choices=["tsne", "umap"])
    g.add_argument("--ncols", type=int, default=4)

    args = ap.parse_args()
    if args.cmd == "single":
        run_single(args)
    else:
        run_grid(args)


if __name__ == "__main__":
    main()

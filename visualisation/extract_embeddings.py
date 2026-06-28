#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Embedding Extractor
===================
Loads a trained DEEPScreen model (CNNModel1/CNNModel2/ViT/YOLOv11) and runs a
forward pass over a data split, capturing the penultimate feature vector (the
input to the final classifier Linear) for every compound. These embeddings are
what t-SNE / UMAP are computed on in plot_projections.py.

It does NOT visualise; it only writes raw data to a .npz, mirroring the two-stage
design of the shap/ tooling.

Output .npz keys
----------------
- 'embeddings'  : (N, D) float32  penultimate features
- 'labels'      : (N,)   int64    1 = active, 0 = inactive (-1 if unknown)
- 'preds'       : (N,)   int64    argmax predicted class
- 'probs'       : (N, 2) float32  softmax probabilities
- 'comp_ids'    : (N,)   str      compound ids
- 'task'        : str    target / task id
- 'model_name'  : str    architecture
- 'split'       : str    split used

Example
-------
python visualisation/extract_embeddings.py \
    --model_path trained_models/.../TASK-CNNModel1-512-256-...-state_dict.pth \
    --task CHEMBL4282 \
    --split test \
    --out_dir visualisation/embeddings \
    --cuda 0

The architecture and (for CNN) fc1/fc2 are auto-parsed from the filename when
present; override with --model_name / --fc1 / --fc2 if needed.
"""

import os
import re
import sys
import json
import argparse
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader

# Make project root importable when run from anywhere
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models import CNNModel1, CNNModel2, ViT, YOLOv11Classifier  # noqa: E402
from data_processing import DEEPScreenDataset  # noqa: E402

warnings.filterwarnings("ignore")

DEFAULT_DATASETS = os.path.join(
    PROJECT_ROOT, "training_files", "target_training_datasets"
)


# -----------------------------------------------------------------------------
# Filename parsing
# -----------------------------------------------------------------------------
def parse_model_filename(path):
    """
    Best-effort parse of the canonical filename:
        TASK_best_val-TASK-<MODELNAME>-<fc1>-<fc2>-<lr>-<bs>-<drop>-<epoch>-...
    Returns dict with any of: model_name, fc1, fc2 that could be inferred.
    """
    name = os.path.basename(path)
    info = {}
    for key in ("CNNModel1", "CNNModel2", "ViT", "YOLOv11"):
        if key in name:
            info["model_name"] = key
            break
    # fc1/fc2 follow the model name: -<MODELNAME>-<fc1>-<fc2>-
    m = re.search(r"-(CNNModel[12])-(\d+)-(\d+)-", name)
    if m:
        info["fc1"] = int(m.group(2))
        info["fc2"] = int(m.group(3))
    return info


# -----------------------------------------------------------------------------
# Model construction
# -----------------------------------------------------------------------------
def build_model(model_name, fc1, fc2, dropout, vit_cfg):
    if model_name == "CNNModel1":
        return CNNModel1(fc1, fc2, dropout)
    if model_name == "CNNModel2":
        return CNNModel2(fc1, fc2, dropout)
    if model_name == "ViT":
        c = vit_cfg
        return ViT(
            c["window_size"], c["hidden_size"], c["attention_probs_dropout_prob"],
            c["drop_path_rate"], dropout, c["layer_norm_eps"], c["encoder_stride"],
            c["embed_dim"], c["depths"], c["mlp_ratio"], num_classes=2,
        )
    if model_name == "YOLOv11":
        return YOLOv11Classifier(num_classes=2, model_size="yolo11m")
    raise ValueError(f"Unknown model_name: {model_name}")


def find_final_linear(model, model_name):
    """
    Return the final classifier nn.Linear whose *input* is the embedding we want.
    """
    if model_name in ("CNNModel1", "CNNModel2"):
        return model.fc3
    if model_name == "ViT":
        # Swinv2ForImageClassification.classifier is a Linear (or Identity if 0 labels)
        clf = model.vit.classifier
        if isinstance(clf, torch.nn.Linear):
            return clf
        # Fallback: last Linear in the module tree
    if model_name == "YOLOv11":
        head = model.model.model[-1]
        if hasattr(head, "linear") and isinstance(head.linear, torch.nn.Linear):
            return head.linear
        if hasattr(head, "fc") and isinstance(head.fc, torch.nn.Linear):
            return head.fc
    # Generic fallback: deepest-registered Linear
    last = None
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            last = m
    if last is None:
        raise RuntimeError("Could not locate a final Linear layer for hooking.")
    return last


# -----------------------------------------------------------------------------
# Extraction
# -----------------------------------------------------------------------------
def extract(model_path, task, split, datasets_path, model_name, fc1, fc2,
            dropout, vit_cfg, batch_size, device):
    print("=" * 70)
    print(f"Task={task}  split={split}  model={model_name}")
    print(f"Model file: {model_path}")
    print(f"Device: {device}")
    print("=" * 70, flush=True)

    model = build_model(model_name, fc1, fc2, dropout, vit_cfg).to(device)
    state = torch.load(model_path, map_location=device)
    # tolerate {'state_dict': ...} wrappers
    if isinstance(state, dict) and "state_dict" in state and not any(
        k.startswith(("conv", "fc", "vit", "model", "bn")) for k in state
    ):
        state = state["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  [warn] missing keys: {len(missing)} (e.g. {missing[:3]})")
    if unexpected:
        print(f"  [warn] unexpected keys: {len(unexpected)} (e.g. {unexpected[:3]})")
    model.eval()

    final_linear = find_final_linear(model, model_name)

    captured = {}

    def pre_hook(module, inputs):
        # inputs[0]: (B, D) features feeding the classifier
        captured["feat"] = inputs[0].detach()

    handle = final_linear.register_forward_pre_hook(pre_hook)

    dataset = DEEPScreenDataset(task, split, parent_path=datasets_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    print(f"Loaded {len(dataset)} samples.", flush=True)

    emb_list, lbl_list, pred_list, prob_list, id_list = [], [], [], [], []

    with torch.no_grad():
        for bi, (imgs, labels, comp_ids) in enumerate(loader, 1):
            imgs = imgs.float().to(device)
            logits = model(imgs)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            feat = captured.get("feat")
            if feat is None:
                raise RuntimeError("Hook did not capture features.")
            feat = feat.reshape(feat.size(0), -1)

            emb_list.append(feat.cpu().numpy().astype(np.float32))
            lbl_list.append(np.asarray(labels).astype(np.int64))
            pred_list.append(preds.cpu().numpy().astype(np.int64))
            prob_list.append(probs.cpu().numpy().astype(np.float32))
            id_list.extend(list(comp_ids))

            if bi % 10 == 0 or bi == len(loader):
                print(f"  batch {bi}/{len(loader)}", flush=True)

    handle.remove()

    embeddings = np.concatenate(emb_list, axis=0)
    labels = np.concatenate(lbl_list, axis=0)
    preds = np.concatenate(pred_list, axis=0)
    probs = np.concatenate(prob_list, axis=0)
    comp_ids = np.asarray(id_list)

    print(f"Embeddings: {embeddings.shape}  | active={int((labels==1).sum())} "
          f"inactive={int((labels==0).sum())}", flush=True)
    return dict(
        embeddings=embeddings, labels=labels, preds=preds, probs=probs,
        comp_ids=comp_ids, task=task, model_name=model_name, split=split,
    )


def main():
    ap = argparse.ArgumentParser(description="Extract penultimate embeddings.")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--task", required=True, help="Target/task folder id, e.g. CHEMBL4282")
    ap.add_argument("--split", default="test",
                    choices=["test", "training", "validation", "all"])
    ap.add_argument("--datasets_path", default=DEFAULT_DATASETS)
    ap.add_argument("--out_dir", default=os.path.join(PROJECT_ROOT, "visualisation", "embeddings"))
    ap.add_argument("--model_name", default=None,
                    choices=[None, "CNNModel1", "CNNModel2", "ViT", "YOLOv11"])
    ap.add_argument("--fc1", type=int, default=None)
    ap.add_argument("--fc2", type=int, default=None)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--cuda", type=int, default=0)
    ap.add_argument("--config", default=os.path.join(PROJECT_ROOT, "config", "config.yaml"),
                    help="YAML with ViT params (used only for ViT).")
    args = ap.parse_args()

    parsed = parse_model_filename(args.model_path)
    model_name = args.model_name or parsed.get("model_name")
    if model_name is None:
        raise SystemExit("Could not infer --model_name from filename; pass it explicitly.")
    fc1 = args.fc1 if args.fc1 is not None else parsed.get("fc1", 128)
    fc2 = args.fc2 if args.fc2 is not None else parsed.get("fc2", 256)

    vit_cfg = {}
    if model_name == "ViT":
        import yaml
        with open(args.config) as f:
            params = yaml.safe_load(f)["parameters"]
        vit_cfg = params

    device = (f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

    result = extract(
        args.model_path, args.task, args.split, args.datasets_path,
        model_name, fc1, fc2, args.dropout, vit_cfg, args.batch_size, device,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{args.task}__{model_name}__{args.split}.npz")
    np.savez_compressed(out_path, **result)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()

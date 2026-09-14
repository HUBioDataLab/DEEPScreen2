#!/usr/bin/env python
"""W&B sweep launcher for TBEV6 (CNN / ViT / YOLO).

Run this on the Anzu GPU server from the DEEPScreen2 repo root, e.g.:

    python run_sweep_tbev6.py --models cnn vit yolo --cuda 6 --count 20

It creates one W&B sweep per model family in the TBEV6 project and starts a
local wandb agent for each. It trains directly from the already-prepared
dataset at training_files/target_training_datasets/TBEV6 (no ChEMBL download,
no re-splitting), reusing train_deepscreen.train_validation_test_training.

Useful flags:
    --dry_run                 print sweep configs and exit (no W&B calls)
    --sweep_ids cnn=ID vit=ID attach agents to existing sweeps instead of
                              creating new ones
    --selection_metric mcc    model-selection metric [auroc, auprc, mcc]
                              (mcc default: historical runs had val recall
                              0.04-0.32 at ~85% accuracy, so AUROC selection
                              ignored the operating point)
    --cuda 6                  CUDA device index (GPU 6 on Anzu)

Diagnosis baked into config/sweep_tbev6_*.yaml (from atabeyunlu/TBEV6):
  - Train ROC AUC ~0.99 vs Validation ~0.73-0.89  -> stronger regularization
    (dropout / att-drop / drop-path), LR decay scheduler, early stopping
  - ViT+AdamW(lr=1e-5) collapsed to all-negative  -> higher LRs explored,
    ViT sweep stays on Muon (best historical run: val ROC 0.893)
  - YOLO fine-tunes from pretrained yolo11m-cls.pt at lower LR
"""

import argparse
import json
import random
import sys
from pathlib import Path

import wandb

REPO_ROOT = Path(__file__).resolve().parent

SWEEP_CONFIGS = {
    "cnn": REPO_ROOT / "config" / "sweep_tbev6_cnn.yaml",
    "vit": REPO_ROOT / "config" / "sweep_tbev6_vit.yaml",
    "yolo": REPO_ROOT / "config" / "sweep_tbev6_yolo.yaml",
}

MODEL_NAME_BY_FAMILY = {
    "cnn": "CNNModel1/CNNModel2",
    "vit": "ViT",
    "yolo": "YOLOv11",
}

SELECTION_METRIC_TO_WANDB = {
    "auroc": "ROC AUC",
    "auprc": "PR AUC",
    "mcc": "MCC",
}


def set_seed(seed):
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_sweep_config(family, selection_metric):
    import yaml

    with open(SWEEP_CONFIGS[family], encoding="utf-8") as f:
        sweep_config = yaml.safe_load(f)

    wandb_metric_name = SELECTION_METRIC_TO_WANDB[selection_metric]
    sweep_config["metric"] = {
        "name": f"Validation/{wandb_metric_name}",
        "goal": "maximize",
    }
    return sweep_config


def check_training_data(training_dir, target_id):
    dataset_path = Path(training_dir) / target_id
    required = [
        dataset_path / "train_val_test_dict.json",
        dataset_path / "imgs",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        sys.exit(
            "Training data not found (this launcher does not download data):\n  "
            + "\n  ".join(missing)
        )
    print(f"Training data found at {dataset_path}")


def make_train_fn(args):
    from train_deepscreen import train_validation_test_training

    def train():
        wandb.init()
        config = dict(wandb.config)
        set_seed(args.run_seed)

        model_name = config["model_name"]
        experiment_name = f"TBEV6_{model_name}_sweep_{wandb.run.id}"
        wandb.run.name = experiment_name

        train_validation_test_training(
            args.target_id,
            model_name,
            config,
            experiment_name,
            args.cuda,
            "None",  # run_id
            "None",  # model_save
            args.project_name,
            args.entity_name,
            args.early_stopping,
            args.patience,
            args.warmup,
            args.selection_metric,
            args.run_seed,
            sweep=True,
            scheduler=args.with_scheduler,
            use_muon=bool(config.get("use_muon", False)),
            split_seed=args.split_seed,
            training_data_root=Path(args.training_dir).resolve(),
        )

    return train


def parse_args():
    parser = argparse.ArgumentParser(description="TBEV6 W&B sweep launcher")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(SWEEP_CONFIGS.keys()),
        default=["cnn", "vit", "yolo"],
        help="Model families to sweep (default: all three)",
    )
    parser.add_argument("--target_id", type=str, default="TBEV6")
    parser.add_argument(
        "--training_dir",
        type=str,
        default="training_files" + chr(47) + "target_training_datasets",
        help="Parent directory containing the TBEV6 dataset folder",
    )
    parser.add_argument("--entity_name", type=str, default="atabeyunlu")
    parser.add_argument("--project_name", type=str, default="TBEV6")
    parser.add_argument(
        "--cuda",
        type=int,
        default=6,
        help="CUDA device index to train on (GPU 6 on Anzu)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=20,
        help="Number of sweep trials per model family",
    )
    parser.add_argument("--split_seed", type=int, default=62)
    parser.add_argument("--run_seed", type=int, default=123)
    parser.add_argument(
        "--early_stopping",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument(
        "--with_scheduler",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--selection_metric",
        type=str,
        default="mcc",
        choices=sorted(SELECTION_METRIC_TO_WANDB.keys()),
        help="Metric used for best-model selection (default: mcc)",
    )
    parser.add_argument(
        "--sweep_ids",
        nargs="*",
        default=[],
        help='Attach to existing sweeps, e.g. cnn=abc123 vit=def456',
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print the sweep configs and exit without touching W&B",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    check_training_data(args.training_dir, args.target_id)

    sweep_ids = {}
    for item in args.sweep_ids or []:
        family, _, sweep_id = item.partition("=")
        if family not in SWEEP_CONFIGS or not sweep_id:
            sys.exit(f"--sweep_ids entry must look like <family>=<id>, got {item!r}")
        sweep_ids[family] = sweep_id

    for family in args.models:
        sweep_config = load_sweep_config(family, args.selection_metric)

        print(f"\n=== Sweep config for {family} ({MODEL_NAME_BY_FAMILY[family]}) ===")
        print(json.dumps(sweep_config, indent=2))

        if args.dry_run:
            continue

        sweep_id = sweep_ids.get(family) or wandb.sweep(
            sweep=sweep_config,
            entity=args.entity_name,
            project=args.project_name,
        )
        print(f"Starting agent for sweep {sweep_id} ({args.count} trials)")
        wandb.agent(
            sweep_id,
            function=make_train_fn(args),
            entity=args.entity_name,
            project=args.project_name,
            count=args.count,
        )


if __name__ == "__main__":
    main()

#!/bin/bash
# TBEV6 sweeps (CNN, ViT, YOLO) on Anzu — GPU 6.
# Run from anywhere; the launcher resolves the repo root itself.
#
#   ./run_tbev6_sweeps_gpu6.sh                 # 20 trials per family
#   ./run_tbev6_sweeps_gpu6.sh --count 10      # fewer trials
#   ./run_tbev6_sweeps_gpu6.sh --models vit    # single family
#
# Requires: wandb logged in (export WANDB_API_KEY=... or `wandb login`),
# dataset at training_files/target_training_datasets/TBEV6/,
# yolo11m-cls.pt in the repo root (auto-downloaded otherwise).

set -euo pipefail
cd "$(dirname "$0")"

python run_sweep_tbev6.py \
    --models cnn vit yolo \
    --cuda 6 \
    --count 20 \
    --selection_metric mcc \
    "$@"

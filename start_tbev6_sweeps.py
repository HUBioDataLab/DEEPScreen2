#!/usr/bin/env python
"""Bootstrap + launch TBEV6 sweeps on Anzu without needing shell activation.

Handles the micromamba env dance in Python:
  1. Locates the `deepscreen` env's interpreter (no `micromamba activate` needed)
  2. Verifies torch + CUDA + wandb credentials + dataset + yolo weights
  3. Re-execs run_sweep_tbev6.py with the env interpreter, forwarding CLI args

Usage on Anzu (works in a bare SSH shell, no hook/init required):

    python3 start_tbev6_sweeps.py                 # all 3 families, GPU 6
    python3 start_tbev6_sweeps.py --models vit    # any run_sweep_tbev6.py flags
    python3 start_tbev6_sweeps.py --check_only    # verify env, don't launch

Run inside tmux (recommended):
    tmux new -s tbev6
    python3 start_tbev6_sweeps.py
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
ENV_NAME = "deepscreen"
SWEEP_LAUNCHER = REPO_ROOT / "run_sweep_tbev6.py"

# Common micromamba root prefixes (first env found wins)
ROOT_PREFIX_CANDIDATES = [
    Path.home() / ".local" / "share" / "mamba-root",
    Path.home() / "micromamba",
    Path.home() / ".micromamba",
    Path.home() / "mamba",
]


def env_python_candidates():
    candidates = []
    for root in ROOT_PREFIX_CANDIDATES:
        candidates.append(root / "envs" / ENV_NAME / "bin" / "python")
    # Also honor an explicitly exported root prefix
    root_env = os.environ.get("MAMBA_ROOT_PREFIX")
    if root_env:
        candidates.append(Path(root_env) / "envs" / ENV_NAME / "bin" / "python")
    # Env dir without a binary might still be usable via a system-level micromamba
    mm = shutil.which("micromamba")
    if mm:
        try:
            out = subprocess.run(
                [mm, "info", "--json"], capture_output=True, text=True, timeout=30
            ).stdout
            import json

            for info in json.loads(out).get("envs", []):
                candidates.append(Path(info) / "bin" / "python")
        except Exception:
            pass
    return [c for c in candidates if c.is_file()]


def pick_python():
    env_pys = env_python_candidates()
    for py in env_pys:
        return py
    return None


def check_python(py):
    print(f"Using interpreter: {py}")
    code = (
        "import torch, wandb;"
        "print('torch', torch.__version__);"
        "print('cuda_available', torch.cuda.is_available());"
        "print('gpu_count', torch.cuda.device_count());"
        "print('wandb', wandb.__version__)"
    )
    result = subprocess.run([str(py), "-c", code], capture_output=True, text=True)
    print(result.stdout.strip())
    if result.returncode != 0:
        print(result.stderr.strip())
        sys.exit(f"Environment check failed for {py}")
    if "cuda_available True" not in result.stdout:
        sys.exit(
            "CUDA is not available in the deepscreen env — "
            "you may be on a login node or the env has a CPU-only torch."
        )


def check_dataset():
    dataset = REPO_ROOT / "training_files" / "target_training_datasets" / "TBEV6"
    split = dataset / "train_val_test_dict.json"
    imgs = dataset / "imgs"
    if not split.is_file() or not imgs.is_dir():
        sys.exit(f"TBEV6 dataset incomplete under {dataset} (need train_val_test_dict.json + imgs/)")
    print(f"Dataset OK: {dataset}")


def check_yolo_weights():
    weights = REPO_ROOT / "yolo11m-cls.pt"
    if weights.is_file():
        print(f"YOLO weights present: {weights}")
    else:
        print("yolo11m-cls.pt not in repo root — ultralytics will auto-download it on first YOLO trial.")


def check_wandb_auth():
    if os.environ.get("WANDB_API_KEY"):
        print("W&B auth: WANDB_API_KEY is set")
        return
    netrc = Path.home() / ".netrc"
    if netrc.exists() and "api.wandb.ai" in netrc.read_text():
        print("W&B auth: found in ~/.netrc")
        return
    print(
        "WARNING: no WANDB_API_KEY env var and no wandb entry in ~/.netrc.\n"
        "The agent will fail if the env is not logged in. "
        "Set it via: export WANDB_API_KEY=... (or run `wandb login` in the env)."
    )


def main():
    check_wandb_auth()
    check_dataset()
    check_yolo_weights()

    py = pick_python()
    if py is None:
        searched = ", ".join(str(c.parent.parent) for c in ROOT_PREFIX_CANDIDATES)
        sys.exit(
            f"Could not find the '{ENV_NAME}' env interpreter. Searched:\n  {searched}\n"
            "Fix by exporting MAMBA_ROOT_PREFIX, or pass the interpreter directly:\n"
            f"  <env>/bin/python run_sweep_tbev6.py --models cnn vit yolo --cuda 6"
        )
    check_python(py)

    if "--check_only" in sys.argv or "--check_only" in [a.split("=")[0] for a in sys.argv]:
        print("Check-only mode: everything looks good, not launching.")
        return

    cmd = [str(py), str(SWEEP_LAUNCHER)] + sys.argv[1:]
    print("\nLaunching:", " ".join(cmd), "\n")
    os.execv(str(py), cmd)  # replace process: clean signal handling under tmux


if __name__ == "__main__":
    main()

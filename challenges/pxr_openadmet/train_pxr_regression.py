import argparse
import os
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.chdir(PROJECT_ROOT)

from train_deepscreen import train_validation_test_training


def main():
    parser = argparse.ArgumentParser(description="Train DEEPScreen2 on the prepared OpenADMET PXR regression dataset")
    parser.add_argument("--target_id", type=str, default="pxr_openadmet")
    parser.add_argument("--model", type=str, default="CNNModel1", choices=["CNNModel1", "ViT", "YOLOv11"])
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--experiment_name", type=str, default="pxr_openadmet_regression")
    parser.add_argument("--project_name", type=str, default="DeepscreenPXR")
    parser.add_argument("--entity_name", type=str, default=None)
    parser.add_argument("--run_id", type=str, default="None")
    parser.add_argument("--model_save", type=str, default="None")
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--selection_metric", type=str, default="rae", choices=["rae", "mae", "r2", "spearman", "kendall"])
    parser.add_argument("--early_stopping", action="store_true")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--with_scheduler", action="store_true")
    parser.add_argument("--muon", action="store_true")
    args = parser.parse_args()

    dataset_dir = PROJECT_ROOT / "training_files" / "target_training_datasets" / args.target_id
    required_files = [
        dataset_dir / "smilesfile.csv",
        dataset_dir / "train_val_test_dict.json",
        dataset_dir / "imgs",
    ]
    missing = [str(p) for p in required_files if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Prepared regression dataset not found for target_id='%s'. Missing: %s" % (args.target_id, missing)
        )

    with open(args.config) as f:
        config = yaml.safe_load(f)["parameters"]

    print(f"Training prepared regression dataset at: {dataset_dir}")
    print(f"Model: {args.model} | Selection metric: {args.selection_metric}")

    train_validation_test_training(
        target_id=args.target_id,
        model_name=args.model,
        config=config,
        experiment_name=args.experiment_name,
        cuda_selection=args.cuda,
        run_id=args.run_id,
        model_save=args.model_save,
        project_name=args.project_name,
        entity=args.entity_name,
        early_stopping=args.early_stopping,
        patience=args.patience,
        warmup=args.warmup,
        selection_metric=args.selection_metric,
        task_type="regression",
        sweep=False,
        scheduler=args.with_scheduler,
        use_muon=args.muon,
    )


if __name__ == "__main__":
    main()

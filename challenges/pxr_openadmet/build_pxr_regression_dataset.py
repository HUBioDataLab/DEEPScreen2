import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_processing import create_regression_training_dataset_from_csv


def main():
    parser = argparse.ArgumentParser(description="Build DEEPScreen2-compatible regression dataset for the OpenADMET PXR challenge")
    parser.add_argument("--input_csv", type=str, default="challenges/pxr_openadmet/pxr-challenge_TRAIN.csv")
    parser.add_argument("--target_id", type=str, default="pxr_openadmet")
    parser.add_argument("--training_dir", type=str, default="training_files/target_training_datasets")
    parser.add_argument("--max_cores", type=int, default=10)
    parser.add_argument("--augment", type=int, default=10)
    parser.add_argument("--scaffold", action="store_true")
    parser.add_argument("--target_column", type=str, default="pEC50")
    parser.add_argument("--smiles_column", type=str, default="SMILES")
    parser.add_argument("--compound_id_column", type=str, default="Molecule Name")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    training_dir = Path(args.training_dir).resolve()
    training_dir.mkdir(parents=True, exist_ok=True)

    metadata = create_regression_training_dataset_from_csv(
        input_csv=args.input_csv,
        target_id=args.target_id,
        target_prediction_dataset_path=str(training_dir),
        max_cores=args.max_cores,
        scaffold=args.scaffold,
        augmentation_angle=args.augment,
        target_column=args.target_column,
        smiles_column=args.smiles_column,
        compound_id_column=args.compound_id_column,
        seed=args.seed,
    )

    print(metadata)


if __name__ == "__main__":
    main()

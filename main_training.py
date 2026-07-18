import argparse
from pathlib import Path
from types import SimpleNamespace
from train_deepscreen import train_validation_test_training
from data_processing import create_final_randomized_training_val_test_sets
from chembl_downloading import download_target
import wandb
import yaml
import os
import time
import random
import numpy as np
import torch

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser(description='DEEPScreen arguments')
# ============================
# General Experiment Settings
# ============================
parser.add_argument(
    '--target_id',
    type=str,
    default=None,
    metavar='TID',
    help='Target/dataset ID (default: config dataset_name or CHEMBL4282)')

parser.add_argument(
    '--dataset', 
    type=str,
    default=None,
    metavar='DATASET',
    help='Dataset format (default: config dataset_format or chembl)')

parser.add_argument(
    '--benchmark',
    action='store_true',
    help='If active, it will run the model across 5 different random seeds for the dataset')

parser.add_argument(
    '--assay_type',
    type=str,
    default='B',
    help="Assay type(s) to search for, comma-separated")

parser.add_argument(
    '--cuda',
    type=int,
    default=0,
    metavar='CORE',
    help='CUDA core index to use (default: 0)')

 
# ============================
# Model & Architecture
# ============================
parser.add_argument(
    '--model',
    type=str,
    default="CNNModel1",
    metavar='MN',
    help='Model name (default: CNNModel1, choices: CNNModel1, CNNModel2, ViT, YOLOv11)')

parser.add_argument(
    '--model_save', 
    type=str, 
    default="None",
    help='Path to previous run if exists (default: None)')

parser.add_argument(
    '--config_folder',
    type=str,
    default="config",
    help="Path to yaml model config files folder")

# ============================
# Data Processing / Augmentation
# ============================

parser.add_argument(
    '--no_fix_tdc',
    action='store_true',
    help='Disable custom fixation for the broken labels in tdc (fixing is enabled by default)')


parser.add_argument(
    '--scaffold',
    action='store_true',
    help='Enable scaffold-based splitting')

parser.add_argument(
    '--split_seed',
    type=int,
    default=0,
    help='Random seed for train/validation/test splitting (default: 0)')

parser.add_argument(
    '--run_seed',
    type=int,
    default=123,
    help='Random seed for model initialization and training (default: 123)')

parser.add_argument(
    '--augment', 
    type=int,
    default=10,
    help='Degrees of rotation for augmentation (default: 10)')

parser.add_argument(
    '--pchembl_threshold',
    type=float,
    default=5.8,
    metavar='DPT',
    help='pChEMBL threshold for selecting data points (default: 5.8)')

parser.add_argument(
    '--similarity_threshold',
    type=float,
    default=50,
    help='Similarity percentage threshold')

parser.add_argument(
    '--negative_enrichment',
    action='store_true',
    help='Enable negative enrichment using similar proteins')


# ============================
# Data Download Options
# ============================
parser.add_argument(
    '--all_proteins',
    action='store_true',
    help="Download data for all protein targets in ChEMBL")

parser.add_argument(
    '--pchembl_threshold_for_download',
    type=int,
    default=0,
    metavar='DPT',
    help='Min. number of datapoints required for download (default: 0)')

parser.add_argument(
    '--output_file',
    type=str,
    default='activity_data.csv',
    help="Output file to save activity data")

parser.add_argument(
    '--training_dir',
    type=str,
    default=f'training_files{os.path.sep}target_training_datasets',
    help='Path to training dataset directory')

parser.add_argument(
    '--smiles_input_file',
    type=str,
    help="Path to txt file containing ChEMBL IDs")

parser.add_argument(
    '--target_process_batch_size',
    type=int, 
    default=10, 
    help="Number of targets to process in each batch")


# ============================
# Subsampling Options
# ============================
parser.add_argument(
    '--subsampling',
    action='store_true',
    help='Enable subsampling to reduce dataset to 3000 samples')

parser.add_argument(
    '--max_total_samples',
    type=int,
    default=3000,
    help='Maximum total samples when subsampling is used (default: 3000)')


# ============================
# Optimization & Scheduling
# ============================
parser.add_argument(
    '--with_scheduler', 
    action='store_true',
    help='Use learning rate scheduler')

parser.add_argument(
    '--muon',
    action='store_true',
    help='Use Muon optimizer (default: Adam)')

parser.add_argument(
    '--early_stopping',
    action='store_true',
    help='Enable early stopping')

parser.add_argument(
    '--patience',
    type=int,
    default=10,
    help='Early stopping patience (epochs)')

parser.add_argument(
    '--warmup',
    type=int,
    default=20,
    help='Epochs to ignore early stopping at the beginning')

parser.add_argument(
    '--selection_metric',
    type=str,
    default='auroc',
    help='Metric used to select the best validation model. Options [auroc, auprc, mcc] (default: auroc)')

# ============================
# Batch & Parallelization
# ============================
parser.add_argument(
    '--max_cores',
    type=int,
    default=10,
    metavar='MAX_CORES',
    help='Maximum number of CPU cores to use')

parser.add_argument(
    '--max_concurrent', 
    type=int, 
    default=50, 
    help="Maximum number of concurrent requests")

# ============================
# Experiment / Logging
# ============================
parser.add_argument(
    '--en',
    type=str,
    default="deepscreen_run",
    metavar='EN',
    help='Experiment name')

parser.add_argument(
    '--project_name', 
    type=str, 
    default='DeepscreenRuns', 
    help="Wandb project name (default: DeepscreenRuns)")

parser.add_argument(
    '--entity_name', 
    type=str, 
    default=None, 
    help="Wandb entity name")

parser.add_argument(
    '--run_id', 
    type=str, 
    default="None",
    help='Wandb Run ID to resume training (default: None)')

parser.add_argument(
    '--sweep', 
    action='store_true',
    help='Enable sweep mode')

parser.add_argument(
    '--email',
    type=str,
    help='E-mail for accessing NCBI BLAST web service')

args = None
DEFAULT_TARGET_ID = "CHEMBL4282"
DEFAULT_DATASET_FORMAT = "chembl"


def resolve_dataset_settings(config_parameters, parsed_args):
    dataset_name = config_parameters.get("dataset_name")
    if parsed_args.target_id is None:
        parsed_args.target_id = dataset_name or DEFAULT_TARGET_ID
    elif dataset_name is not None and dataset_name != parsed_args.target_id:
        raise ValueError(
            f"Config dataset_name '{dataset_name}' does not match "
            f"--target_id '{parsed_args.target_id}'."
        )

    dataset_format = config_parameters.get("dataset_format")
    if parsed_args.dataset is None:
        parsed_args.dataset = dataset_format or DEFAULT_DATASET_FORMAT
    elif dataset_format is not None and dataset_format != parsed_args.dataset:
        raise ValueError(
            f"Config dataset_format '{dataset_format}' does not match "
            f"--dataset '{parsed_args.dataset}'."
        )

    return parsed_args


def sweep(split_seed=None):
    global args
    if split_seed is None:
        split_seed = args.split_seed

    wandb.init(entity = args.entity_name,project=args.project_name, id=args.run_id, resume='allow')

    config = wandb.config
    set_seed(args.run_seed)
    hp_string = "_".join(f"{k}={v}" for k, v in dict(config).items())
    exp_name = f"{args.en}_sweep_{wandb.run.id}_{hp_string}"

    wandb.run.name = exp_name
    print("Batch Size:"+ str(config.bs))

    train_validation_test_training(
        args.target_id,
        args.model,
        config,
        args.en,
        args.cuda,
        args.run_id,
        args.model_save,
        args.project_name,
        args.entity_name,
        args.early_stopping,
        args.patience,
        args.warmup,
        args.selection_metric,
        args.run_seed,
        args.sweep,
        scheduler = args.with_scheduler,
        use_muon = args.muon,
        split_seed=split_seed,
        )



def main():
    global args
    set_seed(args.run_seed)
    repeat = 1
    if args.benchmark: 
        repeat = 5
    config_folder = args.config_folder

    if args.sweep:
        if args.model == "CNNModel2":
            yaml_file = "sweep_cnn2.yaml"
        elif "CNN" in args.model:
            yaml_file = "sweep_cnn.yaml"
        else:
            yaml_file = "sweep_vit.yaml"

        with open(os.path.join(config_folder,yaml_file)) as f:
            sweep_config = yaml.safe_load(f)
        resolve_dataset_settings(sweep_config.get("parameters", {}), args)
    else:
        with open(os.path.join(config_folder,"config.yaml")) as f:
            config = yaml.safe_load(f)
        resolve_dataset_settings(config["parameters"], args)
            
    for seed_offset in range(repeat):
        split_seed = args.split_seed + seed_offset
        print(f"Dataset split seed: {split_seed}")
        # Create platform-independent path
        target_training_dataset_path = Path(args.training_dir).resolve()
        target_training_dataset_path.mkdir(parents=True, exist_ok=True)
        print("start download")
        download_target(args)
        print("end download")
        
        create_final_randomized_training_val_test_sets(
            target_training_dataset_path / args.target_id / args.output_file,
            args.max_cores,
            args.scaffold,
            args.target_id,
            target_training_dataset_path,
            args.dataset,
            args.no_fix_tdc,
            args.pchembl_threshold,
            args.subsampling,
            args.max_total_samples,
            args.similarity_threshold,
            args.negative_enrichment,
            args.augment,
            args.email,
            split_seed)

        if args.sweep:

            sweep_id = wandb.sweep(sweep=sweep_config, project=args.project_name)

            # Start sweep job.
            wandb.agent(sweep_id, function=lambda: sweep(split_seed))
            
        
        else:
            exp_name = args.en
            if args.dataset == "tdc" and args.benchmark:
                exp_name = f"{exp_name}_seed_{split_seed}"

            train_validation_test_training(
            args.target_id,
            args.model,
            config["parameters"],
            exp_name,
            args.cuda,
            args.run_id,
            args.model_save,
            args.project_name,
            args.entity_name,
            args.early_stopping,
            args.patience,
            args.warmup,
            args.selection_metric,
            args.run_seed,
            args.sweep,
            scheduler=args.with_scheduler,
            use_muon = args.muon,
            split_seed=split_seed,
            )
        


if __name__ == "__main__":
    args = parser.parse_args()

    main()
    
    
    

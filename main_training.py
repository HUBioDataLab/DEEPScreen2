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

parser = argparse.ArgumentParser(description='DEEPScreen arguments')
# ============================
# General Experiment Settings
# ============================
parser.add_argument(
    '--target_id',
    type=str,
    default="CHEMBL4282",
    metavar='TID',
    help='Target ChEMBL ID')

parser.add_argument(
    '--dataset', 
    type=str,
    default="chembl",
    metavar='DATASET',
    help='Dataset format (chembl, moleculenet, tdc_adme, tdc_tox)')

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
    help='Model name (default: CNNModel1)')

parser.add_argument(
    '--model_save', 
    type=str, 
    default="None",
    help='Path to previous run if exists (default: None)')

# ============================
# Data Processing / Augmentation
# ============================
parser.add_argument(
    '--scaffold',
    action='store_true',
    help='Enable scaffold-based splitting')

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

parser.add_argument(
    '--batch_size', 
    type=int, 
    default=10, 
    help="Number of targets to process per batch")


# ============================
# Experiment / Logging
# ============================
parser.add_argument(
    '--en',
    type=str,
    default="deepscreen_scaffold_balanced",
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

def sweep():

    wandb.init(entity = args.entity_name,project=args.project_name, id=args.run_id, resume='allow')

    config = wandb.config
    hp_string = "_".join(f"{k}={v}" for k, v in dict(config).items())
    exp_name = f"{args.en}_sweep_{wandb.run.id}_{hp_string}"

    wandb.run.name = exp_name
    wandb.run.save()
    print("Batch Size:"+ str(config.bs))
    train_validation_test_training(
        args.target_id,
        args.model,
        config.fc1,
        config.fc2,
        float(config.learning_rate),
        config.bs,
        config.dropout,
        config.epoch,
        config.hidden_size,
        config.window_size,
        config.attention_probs_dropout_prob,
        config.drop_path_rate,
        config.layer_norm_eps,           
        config.encoder_stride,          
        exp_name,
        args.cuda,
        args.run_id,
        args.model_save,
        args.project_name,
        args.early_stopping,
        args.patience,
        args.warmup,
        args.sweep
        )

def dict_to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    else:
        return d

def main():
    global args
    args = parser.parse_args()

    # Create platform-independent path
    target_training_dataset_path = Path(args.training_dir).resolve()
    target_training_dataset_path.mkdir(parents=True, exist_ok=True)

    download_target(args)

    create_final_randomized_training_val_test_sets(
        target_training_dataset_path / args.target_id / args.output_file,
        args.max_cores,
        args.scaffold,
        args.target_id,
        target_training_dataset_path,
        args.dataset,
        args.pchembl_threshold,
        args.subsampling,
        args.max_total_samples,
        args.similarity_threshold,
        args.negative_enrichment,
        args.augment,
        args.email)
    
    if args.sweep:

        with open("sweep_config.yaml") as f:
            sweep_config = yaml.safe_load(f)

        sweep_id = wandb.sweep(sweep=sweep_config, project=args.project_name)

        # Start sweep job.
        wandb.agent(sweep_id, function=sweep)
        
    
    else:

        
        with open("config.yaml") as f:
            config = yaml.safe_load(f)

        config_ns = dict_to_namespace(config)

        params = config_ns.parameters

        
        train_validation_test_training(
        args.target_id,
        args.model,
        params.fc1,
        params.fc2,
        float(params.learning_rate),
        params.muon_lr,
        params.bs,
        params.dropout,
        params.epoch,
        params.hidden_size,
        params.window_size,
        params.attention_probs_dropout_prob,
        params.drop_path_rate,
        params.layer_norm_eps,           
        params.encoder_stride,
        params.embed_dim,
        params.depths,
        params.mlp_ratio,          
        args.en,
        args.cuda,
        args.run_id,
        args.model_save,
        args.project_name,
        args.entity_name,
        args.early_stopping,
        args.patience,
        args.warmup,
        args.sweep,
        scheduler=args.with_scheduler,
        end_learning_rate_factor=float(params.end_learning_rate),
        use_muon = args.muon
        )
        


if __name__ == "__main__":
    main()
    
    
    

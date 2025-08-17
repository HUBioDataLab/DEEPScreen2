import argparse
from pathlib import Path
from types import SimpleNamespace
from train_deepscreen import train_validation_test_training
from data_processing import create_final_randomized_training_val_test_sets
from chembl_downloading import download_target
import wandb
import yaml

parser = argparse.ArgumentParser(description='DEEPScreen arguments')

# Model and training parameters
parser.add_argument('--target_chembl_id', type=str, default="CHEMBL4282", metavar='TID',
                    help='Target ChEMBL ID')
parser.add_argument('--model', type=str, default="CNNModel1", metavar='MN',
                    help='model name (default: CNNModel1)')
parser.add_argument('--with_scheduler', action='store_true',
                    help='Use learning scheduler(default: False)')
parser.add_argument('--muon', action='store_true',
                    help='Use Muon Optimizer(default: False)')

# Data processing options
parser.add_argument('--scaffold', action='store_true',
                    help='The boolean that controls if the dataset will be spilitted by using scaffold splitting')
parser.add_argument('--en', type=str, default="deepscreen_scaffold_balanced", metavar='EN',
                    help='the name of the experiment (default: my_experiment)')
parser.add_argument('--cuda', type=int, default=0, metavar='CORE',
                    help='The index of cuda core to be used (default: 0)')
parser.add_argument('--pchembl_threshold', type=float, default=6, metavar='DPT',
                    help='The threshold for the number of data points to be used (default: 6)')
parser.add_argument('--moleculenet', action='store_true',
                    help='The boolean that controls if the dataset comes from moleculenet dataset')
parser.add_argument('--all_proteins', action='store_true',
                    help="Download data for all protein targets in ChEMBL")
parser.add_argument('--pchembl_threshold_for_download', type=int, default=0, metavar='DPT',
                    help='The threshold for the number of data points to be used (default: 0)')
parser.add_argument('--assay_type', type=str, default='B',
                    help="Assay type(s) to search for, comma-separated")
parser.add_argument('--max_cores', type=int, default=10, metavar='MAX_CORES',
                    help='Maximum number of CPU cores to use (default: 10)')
parser.add_argument('--output_file', type=str, default='activity_data.csv',
                    help="Output file to save activity data")


# Wandb & logging
parser.add_argument('--project_name', type=str, default='DeepscreenRuns',
                    help="Default project name for wandb runs (default: DeepscreenRuns)")
parser.add_argument('--smiles_input_file', type=str,
                    help="Path to txt file containing ChEMBL IDs")
parser.add_argument('--training_dir', type=str, default='training_files/target_training_datasets',
                    help='Path to training datasets directory (default: training_files/target_training_datasets)')
parser.add_argument('--model_save', type=str, default="None",
                    help='Path to previous run if there exists one (default: None)')
parser.add_argument('--run_id', type=str, default="None",
                    help='Wandb Run ID if you want to continue a run (default: None)')
parser.add_argument('--sweep', action='store_true',
                    help='Sweep mode on or off (default: False)')

args = None

def sweep():
    # wandb'yi başlat (config objesi burada doluyor)

    wandb.init(project=args.project_name, id=args.run_id, resume='allow')

    # Şimdi config güvenle kullanılabilir
    config = wandb.config
    hp_string = "_".join(f"{k}={v}" for k, v in dict(config).items())
    exp_name = f"{args.en}_sweep_{wandb.run.id}_{hp_string}"

    # Deney adını güncelle
    wandb.run.name = exp_name
    wandb.run.save()
    print("Batch Size:"+ str(config.bs))
    train_validation_test_training(
        args.target_chembl_id,
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
        target_training_dataset_path / args.target_chembl_id / args.output_file,
        args.max_cores,
        args.scaffold,
        args.target_chembl_id,
        target_training_dataset_path,
        args.moleculenet,
        args.pchembl_threshold
    )

    
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

        configs = [
            {"name": "0rhhyyru", "encoder_stride": 16, "hidden_size": 600, "layer_norm_eps": 1e-6, "learning_rate": 8e-6, "window_size": 9},
            {"name": "932p7o39", "encoder_stride": 32, "hidden_size": 768, "layer_norm_eps": 1e-4, "learning_rate": 7e-6, "window_size": 7},
            {"name": "n5l3dop4", "encoder_stride": 32, "hidden_size": 1024, "layer_norm_eps": 1e-7, "learning_rate": 1e-5, "window_size": 7},
            {"name": "aiqfnngr", "encoder_stride": 32, "hidden_size": 1024, "layer_norm_eps": 1e-5, "learning_rate": 8e-6, "window_size": 7},
            {"name": "18myoeqw", "encoder_stride": 32, "hidden_size": 1024, "layer_norm_eps": 1e-7, "learning_rate": 8e-6, "window_size": 7},
            {"name": "vzx7oajk", "encoder_stride": 32, "hidden_size": 768, "layer_norm_eps": 1e-6, "learning_rate": 8e-6, "window_size": 7},
            {"name": "f4y80o9r", "encoder_stride": 16, "hidden_size": 600, "layer_norm_eps": 1e-5, "learning_rate": 8e-6, "window_size": 9},
            {"name": "ykzrrhfe", "encoder_stride": 16, "hidden_size": 768, "layer_norm_eps": 1e-5, "learning_rate": 7e-6, "window_size": 7},
        ]

        dropout_configs = [
            {
                "dropout": 0.1,
                "attention_probs_dropout_prob": 0.0,
                "drop_path_rate": 0.05
            },
            {
                "dropout": 0.2,
                "attention_probs_dropout_prob": 0.1,
                "drop_path_rate": 0.1
            },
            {
                "dropout": 0.3,
                "attention_probs_dropout_prob": 0.1,
                "drop_path_rate": 0.1
            },
            {
                "dropout": 0.4,
                "attention_probs_dropout_prob": 0.15,
                "drop_path_rate": 0.15
            },
            {
                "dropout": 0.5,
                "attention_probs_dropout_prob": 0.2,
                "drop_path_rate": 0.2
            }
        ]

        params = config_ns.parameters

        
        for cfg in configs:
            for dropout_set in dropout_configs:

                for i,v in cfg.items():
                    setattr(params, i, v)
                for i,v in dropout_set.items():
                    setattr(params, i, v)

                train_validation_test_training(
                args.target_chembl_id,
                args.model,
                params.fc1,
                params.fc2,
                float(params.learning_rate),
                params.bs,
                params.dropout,
                params.epoch,
                params.hidden_size,
                params.window_size,
                params.attention_probs_dropout_prob,
                params.drop_path_rate,
                params.layer_norm_eps,           
                params.encoder_stride,          
                cfg["name"],
                args.cuda,
                args.run_id,
                args.model_save,
                args.project_name,
                args.sweep,scheduler=args.with_scheduler,
                end_learning_rate_factor=float(params.end_learning_rate),
                use_muon = args.muon
                )


if __name__ == "__main__":
    main()

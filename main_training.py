import argparse
from pathlib import Path
from train_deepscreen import train_validation_test_training
from data_processing import create_final_randomized_training_val_test_sets
from chembl_downloading import download_target
import time

parser = argparse.ArgumentParser(description='DEEPScreen arguments')
parser.add_argument(
    '--target_chembl_id',
    type=str,
    default="CHEMBL4282",
    metavar='TID',
    help='Target ChEMBL ID')
parser.add_argument(
    '--model',
    type=str,
    default="CNNModel1",
    metavar='MN',
    help='model name (default: CNNModel1)')
parser.add_argument(
    '--fc1',
    type=int,
    default=128,
    metavar='FC1',
    help='number of neurons in the first fully-connected layer (default:512)')
parser.add_argument(
    '--fc2',
    type=int,
    default=256,
    metavar='FC2',
    help='number of neurons in the second fully-connected layer (default:256)')
parser.add_argument(
    '--lr',
    type=float,
    default=0.01,
    metavar='LR',
    help='learning rate (default: 0.001)')
parser.add_argument(
    '--bs',
    type=int,
    default=64,
    metavar='BS',
    help='batch size (default: 32)')
parser.add_argument(
    '--dropout',
    type=float,
    default=0.25,
    metavar='DO',
    help='dropout rate (default: 0.25)')
parser.add_argument(
    '--epoch',
    type=int,
    default=2,
    metavar='EPC',
    help='Number of epochs (default: 100)')
parser.add_argument(
    '--scaffold',
    action='store_true',  
    help='The boolean that controls if the dataset will be spilitted by using scaffold splitting')
parser.add_argument(
    '--en',
    type=str,
    default="deepscreen_scaffold_balanced",
    metavar='EN',
    help='the name of the experiment (default: my_experiment)')
parser.add_argument(
    '--cuda',
    type=int,
    default=0,
    metavar='CORE',
    help='The index of cuda core to be used (default: 0)')
parser.add_argument(
    '--pchembl_threshold',
    type=float,  #
    default=5.8,   
    metavar='DPT',
    help='The threshold for the number of data points to be used (default: 6)')
parser.add_argument(
    '--moleculenet', 
    action='store_true',  
    help='The boolean that controls if the dataset comes from moleculenet dataset')
parser.add_argument(
    '--all_proteins',
    action='store_true',
    help="Download data for all protein targets in ChEMBL")
parser.add_argument(
    '--pchembl_threshold_for_download',
    type=int,
    default=0,
    metavar='DPT',
    help='The threshold for the number of data points to be used (default: 0)')
parser.add_argument(
    '--assay_type',
    type=str,
    default='B',
    help="Assay type(s) to search for, comma-separated")
parser.add_argument(
    '--max_cores',
    type=int,
    default=10,
    metavar='MAX_CORES',
    help='Maximum number of CPU cores to use (default: 10)')
parser.add_argument(
    '--output_file',
    type=str,
    default='activity_data.csv',
    help="Output file to save activity data")
parser.add_argument(
    '--smiles_input_file',
    type=str,
    help="Path to txt file containing ChEMBL IDs")
parser.add_argument(
    '--subsampling',
    action='store_true',
    help='Enable subsampling to limit total data points to 3000 with 1:1 positive-negative ratio')
parser.add_argument(
    '--max_total_samples',
    type=int,
    default=3000,
    help='Maximum total number of samples when subsampling is enabled (default: 3000)')
parser.add_argument(
    '--similarity_threshold',
    type=float,
    default=0.5,
    help='Similarity threshold for negative enrichment (default: 0.5 = 50%%)')
parser.add_argument(
    '--negative_enrichment',
    action='store_true',
    help='Enable negative enrichment using similar proteins (requires similarity threshold)')
parser.add_argument(
    '--training_dir',
    type=str,
    default='training_files\\target_training_datasets',
    help='Path to training datasets directory (default: training_files/target_training_datasets)')
parser.add_argument(
    '--max_concurrent', 
    type=int, 
    default=50, 
    help="Maximum number of concurrent requests")
parser.add_argument(
    '--batch_size', 
    type=int, 
    default=10, 
    help="Number of targets to process in each batch")
parser.add_argument(
    '--email',
    type=str,
    help='E-mail adress to access "https://www.ebi.ac.uk/Tools/services/rest/ncbiblast/run')

if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    # Create platform-independent path
    target_training_dataset_path = Path(args.training_dir).resolve()
    # Ensure directory exists
    target_training_dataset_path.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    download_target(args)

    #print(type(args.moleculenet))
    
    
    
    create_final_randomized_training_val_test_sets(
        target_training_dataset_path / args.target_chembl_id / args.output_file,
        args.max_cores,
        args.scaffold,
        args.target_chembl_id,
        target_training_dataset_path,
        args.moleculenet,
        args.pchembl_threshold,
        args.subsampling,
        args.max_total_samples,
        args.similarity_threshold,
        args.negative_enrichment,
        args.email)
    
    train_validation_test_training(args.target_chembl_id, args.model, args.fc1, args.fc2, args.lr, args.bs,
                                   args.dropout, args.epoch, args.en, args.cuda)
    
    
    

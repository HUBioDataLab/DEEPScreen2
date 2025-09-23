import os
import torch
import numpy as np
import pandas as pd
from models import CNNModel1,ViT
from data_processing import save_comp_imgs_from_smiles, initialize_dirs
from torch.utils.data import Dataset, DataLoader
import cv2
import json
from concurrent.futures import ProcessPoolExecutor
import argparse
from tqdm import tqdm

class PredictionDataset(Dataset):
    def __init__(self, target_id, target_prediction_dataset_path, compound_ids):
        self.target_id = target_id
        self.target_prediction_dataset_path = target_prediction_dataset_path
        # Create a list of all possible (compound_id, angle) combinations
        self.samples = [(comp_id, angle) 
                       for comp_id in compound_ids 
                       for angle in range(0, 360, 10)]
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        comp_id, angle = self.samples[index]
        img_path = os.path.join(self.target_prediction_dataset_path, 
                               self.target_id, "imgs", 
                               f"{comp_id}_{angle}.png")
        img_arr = cv2.imread(img_path)
        if img_arr is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        
        img_arr = np.array(img_arr, dtype=np.float32) / 255.0
        img_arr = img_arr.transpose((2, 0, 1))
            
        return img_arr, comp_id

def process_smiles_for_prediction(data):
    smiles, compound_id, target_prediction_dataset_path, target_id = data
    rotations = [(angle, f"_{angle}") for angle in range(0, 360, 10)]
    
    # Check if all rotated images already exist
    all_images_exist = True
    for angle, _ in rotations:
        img_path = os.path.join(target_prediction_dataset_path, 
                               target_id, "imgs", 
                               f"{compound_id}_{angle}.png")
        if not os.path.exists(img_path):
            all_images_exist = False
            break
    
    # Skip if all images exist, otherwise generate them
    if all_images_exist:
        print(f"Skipping {compound_id}: images already exist")
        return compound_id
        
    try:
        save_comp_imgs_from_smiles(target_id, compound_id, smiles, rotations, 
                                 target_prediction_dataset_path)
    except Exception as e:
        print(f"Error processing {compound_id}: {e}")
        return None
    return compound_id

def predict(model_name, model_path, smiles_file, target_id, batch_size=32, cuda_selection=0, fc1=512, fc2=256, dropout=0.1):
    # Setup paths
    current_path_beginning = os.getcwd().split("DEEPScreen")[0]
    current_path_version = os.getcwd().split("DEEPScreen")[1].split("/")[0]
    project_file_path = f"{current_path_beginning}DEEPScreen{current_path_version}"
    target_prediction_dataset_path = f"{project_file_path}/prediction_files"
    
    # Read SMILES file
    df = pd.read_csv(smiles_file)
    smiles_list = df["canonical_smiles"].tolist()
    compound_ids = df["molecule_chembl_id"].tolist()
    
    # Initialize directories
    initialize_dirs(target_id, target_prediction_dataset_path)
    
    # Process SMILES and generate images in parallel
    print("Generating molecule images...")
    smiles_data = [(smiles, comp_id, target_prediction_dataset_path, target_id) 
                   for smiles, comp_id in zip(smiles_list, compound_ids)]
    
    with ProcessPoolExecutor() as executor:
        processed_compounds = list(tqdm(
            executor.map(process_smiles_for_prediction, smiles_data),
            total=len(smiles_data),
            desc="Generating images"
        ))
    processed_compounds = [c for c in processed_compounds if c is not None]
    
    # Setup device
    device = f"cuda:{cuda_selection}" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model
    if model_name == "CNNModel1":
        model = CNNModel1(fc1, fc2, dropout).to(device)
    elif model_name == "ViT":
        model = ViT(num_classes=2, drop_rate=dropout).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Create dataset and dataloader
    dataset = PredictionDataset(target_id, target_prediction_dataset_path, processed_compounds)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Update prediction logic
    predictions = {}
    compound_predictions = {}
    print("Making predictions...")
    with torch.no_grad():
        for batch_imgs, comp_ids in tqdm(dataloader, desc="Predicting"):
            batch_imgs = batch_imgs.to(device)
            outputs = model(batch_imgs)
            probs = torch.argmax(outputs, dim=1) # (batch_size, 2) 
            
            # Accumulate predictions for each compound
            for i, comp_id in enumerate(comp_ids):
                if comp_id not in compound_predictions:
                    compound_predictions[comp_id] = []
                compound_predictions[comp_id].append(probs[i].cpu().item())
    
    # Process final predictions
    for comp_id, rotations in compound_predictions.items():
        active_rotations = sum(rotations)
        predictions[comp_id] = {
            "active_rotations": active_rotations,
            "prediction": 1 if active_rotations >= 18 else 0
        }
    
    # Save predictions
    output_file = f"{target_prediction_dataset_path}/{target_id}/predictions.json"
    with open(output_file, 'w') as f:
        json.dump(predictions, f, indent=2)
    
    print(f"Predictions saved to {output_file}")
    return predictions

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DeepScreen Prediction Script')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to the trained model state dict')
    parser.add_argument('--smiles_file', type=str, required=True,
                      help='Path to CSV file containing SMILES (columns: smiles, compound_id)')
    parser.add_argument('--target_id', type=str, required=True,
                      help='Target ID for prediction')
    parser.add_argument('--batch_size', type=int, default=512,
                      help='Batch size for prediction')
    parser.add_argument('--cuda_selection', type=int, default=1,
                      help='CUDA device index')
    parser.add_argument('--fc1',type=int,default=512,metavar='FC1',
                      help='number of neurons in the first fully-connected layer (default:512)')
    parser.add_argument('--fc2',type=int,default=256,metavar='FC2',
                      help='number of neurons in the second fully-connected layer (default:256)')
    parser.add_argument('--dropout',type=float,default=0.2,metavar='DO',
                      help='dropout rate (default: 0.25)')
    
    args = parser.parse_args()
    predict(args.model_path, args.smiles_file, args.target_id, 
            args.batch_size, args.cuda_selection, args.fc1, 
            args.fc2, args.dropout)
    
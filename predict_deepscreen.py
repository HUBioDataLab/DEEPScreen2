import os
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from models import CNNModel1, ViT
from data_processing import save_comp_imgs_from_smiles, initialize_dirs
from torch.utils.data import Dataset, DataLoader
import cv2
import json
from concurrent.futures import ProcessPoolExecutor
import argparse
import math
from tqdm import tqdm
import yaml
from types import SimpleNamespace
from data_processing import save_comp_imgs_from_smiles, initialize_dirs,get_prediction_loader, load_deepscreen_labels
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score, 
    precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix
)
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
# --- HELPER: Process and Save a Single Heatmap ---

def save_single_heatmap(heatmap_tensor, original_img_tensor, comp_id, output_dir, suffix=""):

    heatmap = heatmap_tensor.numpy()
    heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    original_img = original_img_tensor.permute(1, 2, 0).numpy()
    original_img = (original_img - original_img.min()) / (original_img.max() - original_img.min() + 1e-8)

    # grayscale → RGB convert
    if original_img.shape[2] == 1:
        original_img = np.repeat(original_img, 3, axis=2)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(original_img, alpha=0.6)
    im = ax.imshow(heatmap_norm, cmap="jet", alpha=0.4)
    ax.axis("off")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)

    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Attention Level", rotation=270, labelpad=15)

    save_path = os.path.join(output_dir, f"{comp_id}{suffix}.png")
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)

def process_smiles_for_prediction(data):
    """
    Generates images from SMILES if they don't exist.
    """
    smiles, compound_id, target_prediction_dataset_path, target_id = data
    rotations = [(angle, f"_{angle}") for angle in range(0, 360, 10)]
    
    all_images_exist = True
    for angle, _ in rotations:
        img_path = os.path.join(target_prediction_dataset_path, 
                                target_id, "imgs", 
                                f"{compound_id}_{angle}.png")
        if not os.path.exists(img_path):
            all_images_exist = False
            break
    
    if all_images_exist:
        return compound_id
        
    try:
        save_comp_imgs_from_smiles(target_id, compound_id, smiles, rotations, 
                                 target_prediction_dataset_path)
    except Exception as e:
        print(f"Error processing {compound_id}: {e}")
        return None
    return compound_id

def load_labels(labels_path, target_id=None):
    """
    Loads labels from a JSON file. 
    Supports two formats:
    1. Simple Dict: {"CHEMBL1": 1, "CHEMBL2": 0}
    2. DEEPScreen List: [["CHEMBL1", 1], ["CHEMBL2", 0]]
    """
    if not labels_path or not os.path.exists(labels_path):
        return None
    
    print(f"Loading labels from {labels_path}...")
    try:
        data = json.load(open(labels_path))
        
        # Format 1: Direct Dictionary
        if isinstance(data, dict):
            if "test" in data or "train" in data: # Nested structure
                combined = {}
                for key in data:
                    if isinstance(data[key], list):
                        for item in data[key]:
                            combined[item[0]] = item[1]
                return combined
            elif all(isinstance(v, int) for v in data.values()):
                return data
            
        # Format 2: List of lists
        elif isinstance(data, list):
            label_dict = {}
            for item in data:
                if len(item) >= 2:
                    label_dict[item[0]] = item[1]
            return label_dict
            
    except Exception as e:
        print(f"Warning: Could not parse labels file: {e}")
        return None
    return None

def predict(model_name, model_path, split, target_id, fc1, fc2, batch_size, dropout, hidden_size, window_size, attention_probs_dropout_prob, drop_path_rate, layer_norm_eps, encoder_stride, embed_dim, depths, mlp_ratio, cuda_selection, attention_map_mode):
    # Setup paths
    current_path_beginning = os.getcwd().split("DEEPScreen")[0]
    current_path_version = os.getcwd().split("DEEPScreen")[1].split("/")[0]
    project_file_path = f"{current_path_beginning}DEEPScreen{current_path_version}"
    target_prediction_dataset_path = f"{project_file_path}/prediction_files"
    
    initialize_dirs(target_id, target_prediction_dataset_path)
    
    # Generate Images (if needed) - Assuming process_smiles_for_prediction is called externally or images exist
    # (Skipping generation code here for brevity, assuming images are present)

    # --- LOAD LABELS ---
    label_dict = load_deepscreen_labels(target_id, target_prediction_dataset_path,split)
    has_labels = label_dict is not None
    # Setup Maps
    generate_maps = attention_map_mode in ["all", "avg"]
    maps_output_dir = os.path.join(target_prediction_dataset_path, target_id, "attention_maps")
    if generate_maps and not os.path.exists(maps_output_dir):
        os.makedirs(maps_output_dir)

    device = f"cuda:{cuda_selection}" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load Model
    if model_name == "CNNModel1":
        model = CNNModel1(fc1, fc2, dropout).to(device)
    elif model_name == "ViT":
        model = ViT(window_size, hidden_size, attention_probs_dropout_prob, drop_path_rate, dropout, layer_norm_eps, encoder_stride, embed_dim, depths, mlp_ratio, 2).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()
    
    # Loader
    dataloader = get_prediction_loader(target_id, target_prediction_dataset_path, label_dict, batch_size=batch_size)

    if dataloader is None:
        print("No images found.")
        return
    
    if dataloader is None:
        print("No images found to predict.")
        return

    # Data Storage
    compound_data = {}         
    attention_accumulator = {} 
    
    print(f"Making predictions... (Maps: {attention_map_mode}, Calc Metrics: {has_labels})")
    
    context_manager = torch.enable_grad() if generate_maps else torch.no_grad()
    
    with context_manager:

        for batch_imgs, batch_labels, comp_ids in tqdm(dataloader, desc="Predicting"):
            batch_imgs = batch_imgs.to(device)

            if generate_maps:
                batch_imgs.requires_grad_(True)
                with torch.enable_grad():
                    outputs = model(batch_imgs)
            else:
                with torch.no_grad():
                    outputs = model(batch_imgs)

            # --- Attention Map Logic ---
            if generate_maps:
                score = outputs[:, 1].sum()
                model.zero_grad(set_to_none=True)
                score.backward(retain_graph=False)

                gradients = batch_imgs.grad.data.abs()
                saliency_maps, _ = torch.max(gradients, dim=1)
                saliency_maps = saliency_maps.cpu()
                detached_imgs = batch_imgs.detach().cpu()
                for i, full_id in enumerate(comp_ids):
                    # PARSE ID AND ANGLE
                    # Assumes full_id format is "Name_Angle", e.g., "CHEMBL123_90"
                    # If your ID is "CHEMBL123_90.png", add .replace('.png','')
                    try:
                        if "_" in full_id:
                            base_id = full_id.rsplit('_', 1)[0]
                            angle_str = full_id.rsplit('_', 1)[1]
                            angle = int(angle_str)
                        else:
                            # Fallback if no angle found (assume 0 or single image)
                            base_id = full_id
                            angle = 0
                    except ValueError as e:
                        print(f"Warning: Could not parse angle from {full_id}. Assuming 0.",e)
                        base_id = full_id
                        angle = 0

                    if attention_map_mode == "all":
                        # Save raw map as before
                        save_single_heatmap(saliency_maps[i], detached_imgs[i], full_id, maps_output_dir, suffix="")
                        
                    elif attention_map_mode == "avg":
                        # 1. Get the map for this rotation
                        current_map = saliency_maps[i] # Shape: [H, W]
                        
                        # 2. Un-rotate the map to align with 0-degree view
                        # TF.rotate expects [C, H, W] or PIL, so we unsqueeze to add channel dim
                        if angle != 0:
                            # prepare shape: [N=1, C=1, H, W]
                            current_map = current_map.unsqueeze(0).unsqueeze(0)

                            # build inverse rotation matrix
                            theta = torch.tensor([
                                [ [ math.cos(math.radians(angle)),  math.sin(math.radians(angle)), 0.0 ],
                                [ -math.sin(math.radians(angle)), math.cos(math.radians(angle)), 0.0 ] ]
                            ], dtype=current_map.dtype, device=current_map.device)

                            # create sampling grid for inverse warp
                            grid = F.affine_grid(theta, current_map.size(), align_corners=True)

                            # rotate BACK on GPU
                            current_map = F.grid_sample(current_map, grid, align_corners=True)

                            # restore shape: [H, W]
                            current_map = current_map.squeeze(0).squeeze(0)

                        # 3. Accumulate
                        if base_id not in attention_accumulator:
                            attention_accumulator[base_id] = {
                                'sum_grad': current_map, 
                                'count': 1, 
                                'ref_img': None
                            }
                        else:
                            attention_accumulator[base_id]['sum_grad'] += current_map
                            attention_accumulator[base_id]['count'] += 1
                        
                        # 4. Store the 0-degree image as the reference for the final overlay
                        # We only want to draw the heatmap on top of the upright molecule
                        if angle == 0 or attention_accumulator[base_id]['ref_img'] is None:
                            attention_accumulator[base_id]['ref_img'] = detached_imgs[i]
                            
                del gradients, saliency_maps

            # --- Prediction Logic ---
            soft_probs = F.softmax(outputs.detach(), dim=1)
            batch_preds = torch.argmax(outputs.detach(), dim=1)
            
            batch_preds = batch_preds.cpu()
            soft_probs = soft_probs.cpu()
            batch_labels = batch_labels.cpu()

            for i, full_id in enumerate(comp_ids):
                # Clean the ID for aggregation stats (remove angle suffix)
                if "_" in full_id:
                    clean_id = full_id.rsplit('_', 1)[0]
                else:
                    clean_id = full_id

                if clean_id not in compound_data:
                    compound_data[clean_id] = {"preds": [], "probs": [], "label": -1}
                
                compound_data[clean_id]["preds"].append(batch_preds[i].item())
                compound_data[clean_id]["probs"].append(soft_probs[i][1].item())
                
                current_label = batch_labels[i].item()
                if current_label != -1:
                    compound_data[clean_id]["label"] = current_label
    # --- Post-Processing: Avg Maps ---
    if attention_map_mode == "avg":
        print("Saving averaged maps...")
        for comp_id, data in tqdm(attention_accumulator.items(), desc="Saving"):
            avg_heatmap = data['sum_grad'] / data['count']
            
            # Ensure we have a reference image (fallback if 0-degree was missing)
            ref_img = data['ref_img']
            if ref_img is None:
                print(f"Warning: No 0-degree image found for {comp_id}, skipping map.")
                continue
                
            save_single_heatmap(avg_heatmap, ref_img, comp_id, maps_output_dir, suffix="_avg")

    # --- Aggregation & Metrics ---
    final_predictions = {}
    
    # Lists for Metrics
    y_true = []
    y_pred = []
    y_prob = []
    
    for comp_id, data in compound_data.items():
        rotations_preds = data["preds"]
        rotations_probs = data["probs"]
        total_rotations = len(rotations_preds)
        active_votes = sum(rotations_preds)
        
        # Majority Voting
        is_active = 1 if active_votes >= (total_rotations / 2) else 0
        mean_active_prob = sum(rotations_probs) / total_rotations
        
        confidence = mean_active_prob if is_active == 1 else (1.0 - mean_active_prob)
        
        # Store for JSON
        entry = {
            "prediction": is_active,
            "confidence": confidence,
            "mean_active_probability": mean_active_prob,
            "active_rotations": active_votes,
            "total_rotations": total_rotations
        }
        
        # Check Label for Metrics
        true_label = data["label"]
        if true_label != -1:
            entry["ground_truth"] = true_label
            entry["is_correct"] = (true_label == is_active)
            
            y_true.append(true_label)
            y_pred.append(is_active)
            y_prob.append(mean_active_prob)
            
        final_predictions[comp_id] = entry
    
    # Save Predictions
    output_file = f"{target_prediction_dataset_path}/{target_id}/predictions.json"
    with open(output_file, 'w') as f:
        json.dump(final_predictions, f, indent=2)
    print(f"Predictions saved to {output_file}")

    # --- Calculate Performance Metrics (if labels existed) ---
    if has_labels and len(y_true) > 0:
        print("\n" + "="*30)
        print(f"PERFORMANCE REPORT FOR {target_id} (Split: {split})")
        print("="*30)
        
        try:
            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            mcc = matthews_corrcoef(y_true, y_pred)
            
            try:
                roc_auc = roc_auc_score(y_true, y_prob)
                pr_auc = average_precision_score(y_true, y_prob)
            except ValueError:
                roc_auc = "N/A (One class present)"
                pr_auc = "N/A"
            
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

            metrics = {
                "split": split,
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1-Score": f1,
                "MCC": mcc,
                "ROC AUC": roc_auc,
                "PR AUC": pr_auc,
                "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn)
            }
            
            for k, v in metrics.items():
                print(f"{k}: {v}")
                
            metrics_file = f"{target_prediction_dataset_path}/{target_id}/performance_metrics_{split}.json"
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)
            print(f"Metrics saved to {metrics_file}")
            
        except Exception as e:
            print(f"Error calculating metrics: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DeepScreen Prediction Script')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model state dict')
    parser.add_argument('--target_id', type=str, required=True, help='Target ID for prediction')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for prediction')
    parser.add_argument('--cuda', type=int, default=0, help='CUDA device index')
    parser.add_argument('--split', type=str, default="all", help='Split to predict on (train, validation, test, all)')
    parser.add_argument('--model', type=str, default="CNNModel1", help='Model name (default: CNNModel1)')
    
    # --- ARGUMENT FOR ATTENTION MAPS ---
    parser.add_argument('--attention_map_mode', type=str, default="none", choices=["none", "all", "avg"],
                        help='Mode for attention maps: "none" (default), "all" (save every rotation), or "avg" (save one averaged map per molecule).')
    
    args = parser.parse_args()

    def dict_to_namespace(d):
        if isinstance(d, dict):
            return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
        else:
            return d
        
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    config_ns = dict_to_namespace(config)
    params = config_ns.parameters
    
    predict(
        args.model,
        args.model_path,
        args.split,
        args.target_id,
        params.fc1,
        params.fc2,
        params.bs,
        params.dropout,
        params.hidden_size,
        params.window_size,
        params.attention_probs_dropout_prob,
        params.drop_path_rate,
        params.layer_norm_eps,          
        params.encoder_stride,
        params.embed_dim,
        params.depths,
        params.mlp_ratio,           
        args.cuda,
        args.attention_map_mode
        )
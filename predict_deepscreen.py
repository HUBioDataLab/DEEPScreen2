import os
from pyexpat import model
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from models import CNNModel1, ViT, YOLOv11Classifier
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
from evaluation_metrics import regression_metrics
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable


# örnek kullanım
import matplotlib.pyplot as plt


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
def save_single_heatmap_overlay_attention(
        heatmap, 
        img_tensor, 
        output_path, 
        threshold_ratio=0.85): 
    
    import cv2 
    import matplotlib.pyplot as plt 
    import numpy as np 
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.cm import ScalarMappable

    img = img_tensor.permute(1, 2, 0).detach().cpu().numpy()
    img = np.clip(img, 0, 1)
    h, w = img.shape[:2]

    heatmap_proc = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    heatmap_resized = cv2.resize(heatmap_proc, (w, h), interpolation=cv2.INTER_CUBIC)

    colors = [(1, 1, 1), (1, 0, 0)]
    custom_cmap = LinearSegmentedColormap.from_list("transparent_red", colors)

    mask = (heatmap_resized >= threshold_ratio)
    
    heatmap_grad = np.zeros_like(heatmap_resized)
    heatmap_grad[mask] = (heatmap_resized[mask] - threshold_ratio) / (1.0 - threshold_ratio + 1e-8)

    heatmap_rgb = custom_cmap(heatmap_grad)[:, :, :3]

    # Multiply Blending
    final_overlay = heatmap_rgb * img
    final_overlay = np.clip(final_overlay, 0, 1)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(final_overlay)
    ax.axis('off')

    norm = Normalize(vmin=threshold_ratio, vmax=1)
    sm = ScalarMappable(norm=norm, cmap=custom_cmap)
    sm.set_array([])
    
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Attention Intensity (Thresholded)")

    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def save_single_heatmap_overlay(
        heatmap,
        img_tensor,
        output_path,
        threshold_ratio=0.1,
        dilation_size=5,
        blur_sigma=6):

    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    img = img_tensor.permute(1, 2, 0).detach().cpu().numpy()
    img = np.clip(img, 0, 1)
    h, w = img.shape[:2]

    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    heatmap = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)

    mask = (heatmap > threshold_ratio).astype(np.float32)
    kernel = np.ones((dilation_size, dilation_size), np.uint8)
    expanded_mask = cv2.dilate(mask, kernel, iterations=1)
    expanded_mask = cv2.GaussianBlur(expanded_mask, (0, 0), blur_sigma)
    expanded_mask = np.clip(expanded_mask, 0, 1)

    red_layer = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    white_bg = np.ones((h, w, 3), dtype=np.float32)
    
    heatmap_layer = white_bg * (1 - expanded_mask[..., None]) + red_layer * expanded_mask[..., None]

    final_overlay = heatmap_layer * img
    
    final_overlay = np.clip(final_overlay, 0, 1)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(final_overlay)
    ax.axis("off")

    sm = ScalarMappable(norm=Normalize(vmin=0, vmax=1), cmap="Reds")
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Saliency Intensity", rotation=270, labelpad=15)

    plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

import torch.nn.functional as F
import os, torch, cv2
import numpy as np
import matplotlib.pyplot as plt
def calculate_attn_map(attentions, 
                       img_tensor,):


    B, _, img_h, img_w = img_tensor.shape
    
    rollout_mat = None

    for layer_idx, attn in enumerate(attentions):

        fused_attn = attn.mean(dim=1)
        device = fused_attn.device
        N = fused_attn.size(-1)
        I = torch.eye(N).to(device)
        # fused_attn = 0.5 * fused_attn + 0.5 * I
        
        fused_attn = fused_attn / fused_attn.sum(dim=-1, keepdim=True)

        curr_num_windows_total = fused_attn.shape[0]
        num_windows_per_img = curr_num_windows_total // B

        if rollout_mat is None:
            rollout_mat = fused_attn
        else:
            if rollout_mat.shape[0] != curr_num_windows_total:
                prev_num_windows = rollout_mat.shape[0] // B
                grid_prev = int(np.sqrt(prev_num_windows))
                grid_curr = int(np.sqrt(num_windows_per_img))
                temp_rollout = rollout_mat.view(B, grid_prev, grid_prev, N, N)
                temp_rollout = temp_rollout.permute(0, 3, 4, 1, 2).reshape(B * N * N, 1, grid_prev, grid_prev)
                
                temp_rollout = torch.nn.functional.interpolate(
                    temp_rollout, 
                    size=(grid_curr, grid_curr), 
                    mode='bilinear', 
                    align_corners=False
                )
                
                rollout_mat = temp_rollout.reshape(B, N, N, grid_curr, grid_curr).permute(0, 3, 4, 1, 2).reshape(curr_num_windows_total, N, N)

            rollout_mat = torch.matmul(fused_attn, rollout_mat)

    layer_saliency = rollout_mat.mean(dim=1) 
    
    grid_size = int(np.sqrt(num_windows_per_img))
    window_size = int(np.sqrt(N))
    stitched_map = layer_saliency.view(B, grid_size, grid_size, window_size, window_size)
    stitched_map = stitched_map.permute(0, 1, 3, 2, 4).contiguous()
    stitched_map = stitched_map.view(B, grid_size * window_size, grid_size * window_size)
    return stitched_map

def predict(model_name, model_path, split, target_id, fc1, fc2, batch_size, dropout, hidden_size, window_size, attention_probs_dropout_prob, drop_path_rate, layer_norm_eps, encoder_stride, embed_dim, depths, mlp_ratio, cuda_selection, map_mode, map_type="saliency", task_type="classification"):
    
    current_path_beginning = os.getcwd().split("DEEPScreen")[0]
    current_path_version = os.getcwd().split("DEEPScreen")[1].split("/")[0]
    project_file_path = f"{current_path_beginning}DEEPScreen{current_path_version}"
    target_prediction_dataset_path = f"{project_file_path}/prediction_files"
    
    initialize_dirs(target_id, target_prediction_dataset_path)
    
    # Generate Images (if needed) - Assuming process_smiles_for_prediction is called externally or images exist
    # (Skipping generation code here for brevity, assuming images are present)

    # --- LOAD LABELS ---
    label_dict = load_deepscreen_labels(target_id, target_prediction_dataset_path, split, task_type=task_type)
    has_labels = label_dict is not None
    # Setup Maps
    generate_maps = map_mode in ["all", "avg"]
    maps_output_dir = os.path.join(target_prediction_dataset_path, target_id, "attention_maps")
    if generate_maps and not os.path.exists(maps_output_dir):
        os.makedirs(maps_output_dir)

    device = f"cuda:{cuda_selection}" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load Model
    if model_name == "CNNModel1":
        model = CNNModel1(fc1, fc2, dropout, task_type=task_type).to(device)
    elif model_name == "ViT":
        model = ViT(window_size, hidden_size, attention_probs_dropout_prob, drop_path_rate, dropout, layer_norm_eps, encoder_stride, embed_dim, depths, mlp_ratio, 2, task_type=task_type).to(device)
    elif model_name == "YOLOv11":
        model = YOLOv11Classifier(num_classes=2, model_size="yolo11m", task_type=task_type).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()
    
    # Loader
    dataloader = get_prediction_loader(
        target_id,
        target_prediction_dataset_path,
        label_dict,
        batch_size=batch_size
    )
    
    if dataloader is None:
        print("No images found.")
        return

    # Data Storage
    compound_data = {}         
    attention_accumulator = {} 

    print(f"Making predictions... (Maps: {map_mode}, Calc Metrics: {has_labels})")
        
    print(f"Maps: {map_mode} | Type: {map_type}")
    context = torch.enable_grad() if (generate_maps and map_type == "saliency") else torch.no_grad()

    with context:
        for batch_imgs, batch_labels, comp_ids in tqdm(dataloader):

            batch_imgs = batch_imgs.to(device)

            # -------------------------------------------------
            # FORWARD
            # -------------------------------------------------
            if generate_maps and map_type == "saliency":
                batch_imgs.requires_grad_(True)
                outputs = model(batch_imgs)
            elif generate_maps and map_type == "attention":
                if model_name != "ViT":
                    raise ValueError("Attention rollout only valid for ViT")
                outputs, attentions = model(batch_imgs, return_attention=True)
            else:
                outputs = model(batch_imgs)

            # -------------------------------------------------
            # SALIENCY MAP
            # -------------------------------------------------
            if generate_maps and map_type == "saliency":

                score = outputs[:, 0].sum() if task_type == "regression" else outputs[:, 1].sum()
                model.zero_grad(set_to_none=True)
                score.backward()

                gradients = batch_imgs.grad.data.abs()
                importance_maps, _ = torch.max(gradients, dim=1)
                importance_maps = importance_maps.detach().cpu()
                detached_imgs = batch_imgs.detach().cpu()

            # -------------------------------------------------
            # ATTENTION ROLLOUT
            # -------------------------------------------------
            elif generate_maps and map_type == "attention":

                importance_maps = calculate_attn_map(attentions, batch_imgs)
                importance_maps = importance_maps.detach().cpu()
                detached_imgs = batch_imgs.detach().cpu()

            # -------------------------------------------------
            # MAP SAVE LOGIC
            # -------------------------------------------------
            if generate_maps:

                for i, full_id in enumerate(comp_ids):

                    try:
                        base_id, angle_str = full_id.rsplit("_", 1)
                        angle = int(angle_str)
                    except:
                        base_id, angle = full_id, 0

                    current_map = importance_maps[i]

                    # Resize to image resolution
                    img_h, img_w = detached_imgs[i].shape[1:]
                    current_map = F.interpolate(
                        current_map.unsqueeze(0).unsqueeze(0),
                        size=(img_h, img_w),
                        mode="bilinear",
                        align_corners=True
                    ).squeeze()

                    if map_mode == "all":
                        out_p = os.path.join(maps_output_dir, f"{full_id}.png")
                        if map_type == "saliency":
                            save_single_heatmap_overlay(
                                current_map.numpy(),
                                detached_imgs[i],
                                out_p
                            )
                        else:
                            save_single_heatmap_overlay_attention(
                                current_map.numpy(),
                                detached_imgs[i],
                                out_p
                            )

                    elif map_mode == "avg":

                        if angle != 0:
                            rotated_map = TF.rotate(
                                current_map.unsqueeze(0),
                                -angle,
                                interpolation=TF.InterpolationMode.BILINEAR
                            ).squeeze()
                        else:
                            rotated_map = current_map

                        if base_id not in attention_accumulator:
                            attention_accumulator[base_id] = {
                                "sum_map": rotated_map.numpy(),
                                "count": 1,
                                "ref_img": detached_imgs[i] if angle == 0 else None
                            }
                        else:
                            attention_accumulator[base_id]["sum_map"] += rotated_map.numpy()
                            attention_accumulator[base_id]["count"] += 1
                            if angle == 0:
                                attention_accumulator[base_id]["ref_img"] = detached_imgs[i]

            # -------------------------------------------------
            # PREDICTION
            # -------------------------------------------------
            if task_type == "regression":
                batch_preds = outputs.detach().view(-1)
            else:
                soft_probs = F.softmax(outputs.detach(), dim=1)
                batch_preds = torch.argmax(outputs.detach(), dim=1)

            for i, full_id in enumerate(comp_ids):

                clean_id = full_id.rsplit("_", 1)[0] if "_" in full_id else full_id

                if clean_id not in compound_data:
                    compound_data[clean_id] = {"preds": [], "probs": [], "label": -1}

                if task_type == "regression":
                    compound_data[clean_id]["preds"].append(float(batch_preds[i].item()))
                else:
                    compound_data[clean_id]["preds"].append(batch_preds[i].item())
                    compound_data[clean_id]["probs"].append(soft_probs[i][1].item())

                label = batch_labels[i].item()
                if label != -1:
                    compound_data[clean_id]["label"] = label

            
    # -------------------- AVG SAVE --------------------
    if generate_maps and map_mode == "avg":
        for cid, data in attention_accumulator.items():
            final_avg = data["sum_map"] / data["count"]
            ref = data["ref_img"]
            if ref is None:
                continue

            out_p = os.path.join(maps_output_dir, f"{cid}_avg.png")
            if map_type == "saliency":
                save_single_heatmap_overlay(final_avg, ref, out_p)
            else:
                save_single_heatmap_overlay_attention(final_avg, ref, out_p)
                
    print("Prediction finished.")

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

        if task_type == "regression":
            pred_value = float(sum(rotations_preds) / total_rotations)
            entry = {
                "prediction": pred_value,
                "total_rotations": total_rotations
            }
        else:
            active_votes = sum(rotations_preds)
            is_active = 1 if active_votes >= (total_rotations / 2) else 0
            mean_active_prob = sum(rotations_probs) / total_rotations
            confidence = mean_active_prob if is_active == 1 else (1.0 - mean_active_prob)
            entry = {
                "prediction": is_active,
                "confidence": confidence,
                "mean_active_probability": mean_active_prob,
                "active_rotations": active_votes,
                "total_rotations": total_rotations
            }
        
        true_label = data["label"]
        if true_label != -1:
            entry["ground_truth"] = true_label
            if task_type == "regression":
                entry["absolute_error"] = abs(true_label - pred_value)
                y_true.append(true_label)
                y_pred.append(pred_value)
            else:
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
            if task_type == "regression":
                metrics = {"split": split, **regression_metrics(y_true, y_pred)}
            else:
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
    parser.add_argument('--task_type', type=str, default='classification', choices=['classification', 'regression'], help='Prediction head/task type.')
    
    # --- ARGUMENT FOR ATTENTION MAPS ---
    parser.add_argument('--map_mode', type=str, default="none", choices=["none", "all", "avg"],
                        help='Mode for attention maps: "none" (default), "all" (save every rotation), or "avg" (save one averaged map per molecule).')
    parser.add_argument(
        '--map_type',
        type=str,
        default="saliency",
        choices=["saliency", "attention"],
        help="saliency (grad-based) or attention (ViT rollout)"
    )
    args = parser.parse_args()

    def dict_to_namespace(d):
        if isinstance(d, dict):
            return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
        else:
            return d
        
    with open("config/config.yaml") as f:
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
        args.map_mode,
        args.map_type,
        args.task_type,
        )
import os

import torch

import warnings

import numpy as np

import torch.nn as nn
from torch.optim.lr_scheduler import LinearLR

from models import CNNModel1, ViT,YOLOv11Classifier

from data_processing import get_train_test_val_data_loaders
from evaluation_metrics import prec_rec_f1_acc_mcc, get_list_of_scores

from sklearn.metrics import roc_auc_score, average_precision_score

from data_processing import generate_images, save_comp_imgs_from_smiles

import wandb

import matplotlib.pyplot as plt
import cv2
import shutil

from muon import SingleDeviceMuonWithAuxAdam

warnings.filterwarnings(action='ignore')
torch.manual_seed(123)
np.random.seed(123)
use_gpu = torch.cuda.is_available()

sep = os.sep

current_path_beginning = os.getcwd().split("DEEPScreen")[0]
current_path_version = os.getcwd().split("DEEPScreen")[1].split("/")[0]

project_file_path = f"{current_path_beginning}DEEPScreen{current_path_version}"
training_files_path = f"{project_file_path}{sep}training_files"
result_files_path = f"{project_file_path}{sep}result_files"
trained_models_path = f"{project_file_path}{sep}trained_models"



def save_best_model_predictions(experiment_name, epoch, validation_scores_dict, test_scores_dict, model, project_file_path, target_id, str_arguments,
                                all_test_comp_ids, test_labels, test_predictions,global_step,optimizer):

    if not os.path.exists(os.path.join(trained_models_path, experiment_name)):
            os.makedirs(os.path.join(trained_models_path, experiment_name))
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'steps': global_step,
            }, os.path.join(trained_models_path,experiment_name, target_id+"_"+ str_arguments+'-checkpoint.pth'))

    torch.save(model.state_dict(),
               os.path.join(trained_models_path,experiment_name,target_id+"_best_val-"+str_arguments+"-state_dict.pth"))
    
    str_test_predictions = "CompoundID\tLabel\tPred\n"
    for ind in range(len(all_test_comp_ids)):
        str_test_predictions += "{}\t{}\t{}\n".format(all_test_comp_ids[ind],
                                                          test_labels[ind],
                                                          test_predictions[ind])
    best_test_performance_dict = test_scores_dict
    best_test_predictions = str_test_predictions
    return validation_scores_dict, best_test_performance_dict, best_test_predictions, str_test_predictions

def get_device(cuda_selection):
    device = "cpu"
    if use_gpu:
        print("GPU is available on this device!")
        device = "cuda:"+str(cuda_selection)
    else:
        print("CPU is available on this device!")
    return device

def calculate_val_test_loss(model, criterion, data_loader, device):
    total_count = 0
    total_loss = 0.0
    all_comp_ids = []
    all_labels = []
    all_predictions = []
    all_pred_probs = []

    for i, data in enumerate(data_loader):

        img_arrs, labels, comp_ids = data
        img_arrs, labels = torch.tensor(img_arrs).type(torch.FloatTensor).to(device), torch.tensor(labels).to(device)
        total_count += len(comp_ids)
        y_pred = model(img_arrs).to(device)
        loss = criterion(y_pred, labels)
        total_loss += float(loss.item())
        all_comp_ids.extend(list(comp_ids))
        _, preds = torch.max(y_pred, 1)
        all_labels.extend(list(labels.detach().cpu().numpy()))
        all_predictions.extend(list(preds.detach().cpu().numpy()))
        all_pred_probs.extend(y_pred.detach().cpu().numpy())


    return total_loss, total_count, all_comp_ids, all_labels, all_predictions,all_pred_probs

def aggregate_predictions(comp_ids, labels, predictions, pred_probs):

    unique_mols = {}
    
    # Collect molecules in dict
    for cid, lab, pred, prob in zip(comp_ids, labels, predictions, pred_probs):
        cid = cid.rsplit('_', 1)[0]
        if cid not in unique_mols.keys():
            unique_mols[cid] = {"labels": [], "preds": [], "probs": []}
        unique_mols[cid]["labels"].append(lab)
        unique_mols[cid]["preds"].append(pred)
        unique_mols[cid]["probs"].append(prob)
    agg_labels = []
    agg_preds = []
    agg_probs = []
    agg_comp_ids = []
    
    for cid, data in unique_mols.items():
        # Label: All of are the same so we can get the first
        current_label = data["labels"][0]
        
        # Voting 
        total_rotations = len(data["preds"])
        positive_votes = sum(data["preds"]) 
        
        if positive_votes >= (total_rotations / 2):
            final_pred = 1
        else:
            final_pred = 0
            
        avg_prob = np.mean(np.array(data["probs"]), axis=0)

        agg_comp_ids.append(cid)
        agg_labels.append(current_label)
        agg_preds.append(final_pred)
        agg_probs.append(avg_prob)
    return agg_comp_ids, agg_labels, agg_preds, agg_probs

def train_validation_test_training(
    target_id, model_name, config, experiment_name, cuda_selection, run_id, model_save, project_name, entity,
    
    early_stopping, 
    patience, 
    warmup,
    
    sweep=False, scheduler=False, use_muon=False
):

    # ---- 1. CONFIGURATION MERGE (Fail-safe) ----
    # Add runtime/function args to cfg so they are logged too
    cfg = {
        "target_id": target_id,
        "model_name": model_name,
        "experiment_name": experiment_name,
        "cuda_selection": cuda_selection,
        "scheduler": scheduler,
        "use_muon": use_muon,
        "model_save": model_save
    }
    
    for i,v in config.items():
        cfg[i] = v
    # ---- 2. GENERATE RUN STRING ----
    # Create a unique identifier string based on key hyperparameters
    # We select specific keys to be part of the filename/ID
    key_params = [
        target_id, model_name, cfg['fc1'], cfg['fc2'], 
        cfg['learning_rate'], cfg['bs'], cfg['dropout'], cfg['epoch'], 
        experiment_name
    ]
    
    # Clean formatting for floats (removes trailing zeros)
    arguments_list = [
        "{:.16f}".format(x).rstrip('0').rstrip('.') if isinstance(x, float) else str(x)
        for x in key_params
    ]
    str_arguments = "-".join(arguments_list)
    print("Run ID:", str_arguments)

    # ---- 3. W&B LOGGING ----
    if not sweep:
        wandb_args = {
            "project": project_name,
            "id": run_id if run_id != "None" else None,
            "name": experiment_name,
            "resume": "allow",
            "config": cfg, # Just pass the whole merged dict!
        }
        
        if entity not in [None, "", "None"]:
            wandb_args["entity"] = entity

        wandb.init(**wandb_args)

    # ---- 4. INITIALIZATION ----
    device = get_device(cuda_selection)
    
    # Setup Paths
    exp_path = os.path.join(result_files_path, "experiments", experiment_name)
    os.makedirs(exp_path, exist_ok=True) # exist_ok=True replaces the if check
    os.makedirs(os.path.join(trained_models_path, experiment_name), exist_ok=True)

    # File Handlers (Using 'with' is better usually, but keeping your style for persistence)
    # Note: Added 'str_arguments' to filename as requested
    res_file_path = os.path.join(exp_path, f"best_val_test_performance_results-{str_arguments}.txt")
    pred_file_path = os.path.join(exp_path, f"best_val_test_predictions-{str_arguments}.txt")
    
    best_val_test_result_fl = open(res_file_path, "w")
    best_val_test_prediction_fl = open(pred_file_path, "w")

    # Data Loaders
    train_loader, valid_loader, test_loader = get_train_test_val_data_loaders(target_id, cfg['bs'])

    # ---- 5. DYNAMIC MODEL LOADING ----
    # This is the "Bugless" part. We map model names to classes and specific args.
    model = None
    
    if model_name == "CNNModel1":
        model = CNNModel1(
            cfg['fc1'], 
            cfg['fc2'], 
            cfg['dropout']
        ).to(device)
        
    elif model_name == "ViT":
        # For complex models, you can pass parameters explicitly or using **cfg
        # if the model arguments match your dictionary keys exactly.
        model = ViT(
            cfg['window_size'],
            cfg['hidden_size'],
            cfg['attention_probs_dropout_prob'],
            cfg['drop_path_rate'],
            cfg['dropout'],
            cfg['layer_norm_eps'],
            cfg['encoder_stride'],
            cfg['embed_dim'],
            cfg['depths'],
            cfg['mlp_ratio'],
            num_classes=2
        ).to(device)

    elif model_name == "YOLOv11":
        
        model = YOLOv11Classifier(num_classes=2,model_size="yolo11m").to(device)
        
    else:
        raise ValueError(f"Model '{model_name}' is not recognized.")
    # ---- 6. OPTIMIZER, SCHEDULER, CRITERION ----
    n_epoch = cfg['epoch']
    learning_rate = float(cfg['learning_rate'])
    muon_lr = float(cfg.get('muon_lr')) 
    end_learning_rate_factor = cfg.get('end_learning_rate_factor', None)
    

    if use_muon: 
        if model_name=="ViT":
            hidden_weights = [p for p in model.vit.swinv2.encoder.parameters() if p.ndim >= 2]
            hidden_gains_biases = [p for p in model.vit.swinv2.encoder.parameters() if p.ndim < 2]
            nonhidden_params = [*model.vit.swinv2.embeddings.parameters(), *model.vit.classifier.parameters(),*model.vit.swinv2.layernorm.parameters()]
            param_groups = [
                dict(params=hidden_weights, use_muon=True,
                    lr=muon_lr),
                dict(params=hidden_gains_biases+nonhidden_params, use_muon=False,
                    lr=learning_rate, betas=(0.9, 0.95),),
            ]

        elif model_name == "YOlOv11":

            backbone_neck = model.model[:-1]
            head = model.model[-1]

            hidden_weights = []
            hidden_gains_biases = []

            for m in backbone_neck:
                for p in m.parameters():
                    if p.ndim >= 2:
                        hidden_weights.append(p)
                    else:
                        hidden_gains_biases.append(p)

            head_params = list(head.parameters())

            param_groups = [
                dict(
                    params=hidden_weights,
                    use_muon=True,
                    lr=muon_lr
                ),
                dict(
                    params=hidden_gains_biases + head_params,
                    use_muon=False,
                    lr=learning_rate,
                    betas=(0.9, 0.95)
                )
            ]

            optimizer = SingleDeviceMuonWithAuxAdam(param_groups)



        else:

            hidden_weights = [p for p in model.parameters() if p.ndim >= 2]
            
            hidden_gains_biases = [p for p in model.parameters() if p.ndim < 2]

            param_groups = [
                dict(params=hidden_weights, use_muon=True, 
                     lr=muon_lr), 
                
                dict(params=hidden_gains_biases, use_muon=False, 
                     lr=learning_rate, betas=(0.9, 0.95)),
            ]
        optimizer = SingleDeviceMuonWithAuxAdam(param_groups)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    if model_save!="None":
        checkpoint = torch.load(model_save)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        start_step = checkpoint['steps'] + 1
    else:
        start_epoch = 0
        start_step = 0
    
    if scheduler:
        if not end_learning_rate_factor:
            end_learning_rate_factor = 0.2
        scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=end_learning_rate_factor, total_iters=n_epoch)
            

    criterion = nn.CrossEntropyLoss()
    optimizer.zero_grad()

    best_val_mcc_score, best_test_mcc_score = 0.0, 0.0
    best_val_test_performance_dict = dict()
    best_val_test_performance_dict["MCC"] = 0.0

    best_test_performance_dict = {}
    best_test_predictions = ""

    best_val_roc_auc = 0.0

    early_stopping_counter = 0

    global_step = start_step  # Track steps across all epochs


    for e in range(n_epoch):
        epoch = e + start_epoch
        total_training_count = 0
        total_training_loss = 0.0
        print("Epoch :{}".format(epoch))
        model.train()
        batch_number = 0
        all_training_labels = []
        all_training_preds = []
        all_training_probs = []
        print("Training mode:", model.training)

        for i, data in enumerate(train_loader):
            batch_number += 1
            optimizer.zero_grad()
            img_arrs, labels, comp_ids = data
            img_arrs = torch.tensor(img_arrs).type(torch.FloatTensor).to(device)
            labels = torch.tensor(labels).to(device)

            total_training_count += len(comp_ids)

            y_pred = model(img_arrs).to(device)
            _, preds = torch.max(y_pred, 1)
            all_training_labels.extend(list(labels.detach().cpu().numpy()))
            all_training_preds.extend(list(preds.detach().cpu().numpy()))
            all_training_probs.extend(y_pred.detach().cpu().numpy())

            loss = criterion(y_pred, labels)
            total_training_loss += float(loss.item())
            loss.backward()
            optimizer.step()

           
            # Wandb log at every step
            wandb.log({"Loss/train_step": loss.item(), "step": global_step})

            global_step += 1  # Increment global step

        if scheduler:
            scheduler.step()
 
        print("Epoch {} training loss:".format(epoch), total_training_loss)
        
        wandb.log({"Loss/train": total_training_loss, "epoch": epoch})
        
        training_perf_dict = dict()
        try:
            training_perf_dict = prec_rec_f1_acc_mcc(all_training_labels, all_training_preds)
        except Exception as e:
            print(f"Problem during training performance calculation! {e}")
        
        training_roc_auc = roc_auc_score(all_training_labels, np.array(all_training_probs)[:, 1])
        training_pr_auc = average_precision_score(all_training_labels, np.array(all_training_probs)[:, 1])
        training_perf_dict["ROC AUC"] = training_roc_auc
        training_perf_dict["PR AUC"] = training_pr_auc

        for metric, value in training_perf_dict.items():
            wandb.log({f"Train/{metric}": value, "epoch": epoch})

        model.eval()

        # Save the model checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'steps': global_step,
        }, os.path.join(trained_models_path, experiment_name, target_id + "_" + str_arguments + '-checkpoint.pth'))

        with torch.no_grad():
            print("Validation mode:", not model.training)

            # --- VALIDATION ---
            total_val_loss, total_val_count, raw_val_comp_ids, raw_val_labels, raw_val_predictions, raw_val_probs = calculate_val_test_loss(model, criterion, valid_loader, device)

            all_val_comp_ids, all_val_labels, val_predictions, val_pred_probs = aggregate_predictions(
                raw_val_comp_ids, raw_val_labels, raw_val_predictions, raw_val_probs
            )
            
            val_perf_dict = dict()
            val_perf_dict["MCC"] = 0.0
            val_predictions_tensor = torch.tensor(val_predictions)
            all_val_labels_tensor = torch.tensor(all_val_labels)
            
            try:
                val_perf_dict = prec_rec_f1_acc_mcc(all_val_labels_tensor, val_predictions_tensor)
            except Exception as e:
                print(f"There was a d during validation performance calculation: {e}")

            
            val_roc_auc = roc_auc_score(all_val_labels, np.array(val_pred_probs)[:, 1])
            val_pr_auc = average_precision_score(all_val_labels, np.array(val_pred_probs)[:, 1])
            val_perf_dict["ROC AUC"] = val_roc_auc
            val_perf_dict["PR AUC"] = val_pr_auc

            for metric, value in val_perf_dict.items():
                wandb.log({f"Validation/{metric}": value, "epoch": epoch})

            total_test_loss, total_test_count, raw_test_comp_ids, raw_test_labels, raw_test_predictions, raw_test_probs = calculate_val_test_loss(
                model, criterion, test_loader, device)

            all_test_comp_ids, all_test_labels, test_predictions, test_pred_probs = aggregate_predictions(
                raw_test_comp_ids, raw_test_labels, raw_test_predictions, raw_test_probs
            )

            test_perf_dict = dict()
            test_perf_dict["MCC"] = 0.0
            test_predictions_tensor = torch.tensor(val_predictions)
            all_test_labels_tensor = torch.tensor(all_val_labels)
            try:
                test_perf_dict = prec_rec_f1_acc_mcc(all_test_labels_tensor, test_predictions_tensor)
            except Exception as e:
                print(f"There was a problem during test performance calculation: {e}")

            test_roc_auc = roc_auc_score(all_test_labels, np.array(test_pred_probs)[:, 1])
            test_pr_auc = average_precision_score(all_test_labels, np.array(test_pred_probs)[:, 1])
            test_perf_dict["ROC AUC"] = test_roc_auc
            test_perf_dict["PR AUC"] = test_pr_auc

            for metric, value in test_perf_dict.items():
                wandb.log({f"Test/{metric}": value, "epoch": epoch})


            if early_stopping and epoch >= warmup:
                print("Val ROC AUC score: ",val_perf_dict["ROC AUC"])
                print("Best val ROC AUC: ", best_val_roc_auc)

                current_val_roc_auc_2digit = round(val_perf_dict["ROC AUC"], 2)
                best_val_roc_auc_2digit = round(best_val_roc_auc, 2)

                improved = current_val_roc_auc_2digit > best_val_roc_auc_2digit
                if improved:
                    early_stopping_counter = 0
                    print("Early stopping number : ", early_stopping_counter)
                else:
                    early_stopping_counter += 1
                    print("Early stopping number : ", early_stopping_counter)
                    if early_stopping_counter >= patience:


                        wandb.log({"Loss/validation": total_val_loss, "epoch": epoch})
                        wandb.log({"Loss/test": total_test_loss, "epoch": epoch})

                        
                        score_list = get_list_of_scores()
                        for scr in score_list:
                            best_val_test_result_fl.write("Test {}:\t{}\n".format(scr, best_test_performance_dict[scr]))
                        best_val_test_prediction_fl.write(best_test_predictions)

                        best_val_test_result_fl.close()
                        best_val_test_prediction_fl.close()


                        print(f"Early stopping triggered at epoch {epoch}")
                        break


            if val_perf_dict["ROC AUC"] > best_val_roc_auc:

                best_val_roc_auc = val_perf_dict["ROC AUC"]


            if val_perf_dict["MCC"] > best_val_mcc_score:
                best_val_mcc_score = val_perf_dict["MCC"]
                best_test_mcc_score = test_perf_dict["MCC"]
                
                validation_scores_dict, best_test_performance_dict, best_test_predictions, str_test_predictions = save_best_model_predictions(
                    experiment_name, epoch, val_perf_dict, test_perf_dict,
                    model,project_file_path, target_id, str_arguments,
                    all_test_comp_ids, all_test_labels, test_predictions, global_step,optimizer)
            
        wandb.log({"Loss/validation": total_val_loss, "epoch": epoch})
        wandb.log({"Loss/test": total_test_loss, "epoch": epoch})

        if epoch == n_epoch - 1:
            score_list = get_list_of_scores()
            for scr in score_list:
                best_val_test_result_fl.write("Test {}:\t{}\n".format(scr, best_test_performance_dict[scr]))
            best_val_test_prediction_fl.write(best_test_predictions)

            best_val_test_result_fl.close()
            best_val_test_prediction_fl.close()
        
        
    
    wandb.finish()


import os

import torch

import warnings

import numpy as np

import torch.nn as nn
from torch.optim.lr_scheduler import LinearLR

from models import CNNModel1, ViT

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

def train_validation_test_training(target_id, model_name, fully_layer_1, fully_layer_2, learning_rate, batch_size, drop_rate, n_epoch,hidden_size,window_size,att_drop,drop_path_rate,layer_norm_eps,encoder_stride, embed_dim,depths,mlp_ratio , experiment_name, cuda_selection,run_id,model_save,project_name,sweep=False,scheduler = False,end_learning_rate_factor = None,use_muon=False):

    arguments = ["{:.16f}".format(argm).rstrip('0') if type(argm)==float else str(argm) for argm in
                 [target_id, model_name, fully_layer_1, fully_layer_2, learning_rate, batch_size, drop_rate, n_epoch, experiment_name]]

    best_test_performance_dict = {"Accuracy": 0, "Precision": 0, "Recall": 0, "F1-Score": 0, "MCC": 0, "TP": 0, "TN": 0, "FP": 0, "FN": 0}
    best_test_predictions = ""

    str_arguments = "-".join(arguments)
    print("Arguments:", str_arguments)
    
    if run_id =="None":
        run_id = None

    if not sweep:

        wandb.init(project=project_name, id = run_id,name=experiment_name,resume = 'allow', config={
            "target_id": target_id,
            "model_name": model_name,
            "fully_layer_1": fully_layer_1,
            "fully_layer_2": fully_layer_2,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "drop_rate": drop_rate,
            "n_epoch": n_epoch,
            "cuda_selection": get_device(cuda_selection)
        })


    device = get_device(cuda_selection)
    exp_path = os.path.join(result_files_path, "experiments", experiment_name)

    if not os.path.exists(exp_path):
        os.makedirs(exp_path)

    if not os.path.exists(os.path.join(trained_models_path, experiment_name)):
        os.makedirs(os.path.join(trained_models_path, experiment_name))

    best_val_test_result_fl = open(
        os.path.join(exp_path,"best_val_test_performance_results-"+str_arguments+".txt"), "w")
    best_val_test_prediction_fl = open(
        os.path.join(exp_path,"best_val_test_predictions-"+str_arguments+".txt"), "w")

    train_loader, valid_loader, test_loader = get_train_test_val_data_loaders(target_id, batch_size)
    
    model = None
    if model_name == "CNNModel1":
        model = CNNModel1(fully_layer_1, fully_layer_2, drop_rate).to(device)
    elif model_name == "ViT":
        model = ViT(window_size,hidden_size,att_drop,drop_path_rate,drop_rate,layer_norm_eps,encoder_stride,embed_dim,depths,mlp_ratio,2).to(device)

    if use_muon and model_name=="ViT":
        hidden_weights = [p for p in model.vit.swinv2.encoder.parameters() if p.ndim >= 2]
        hidden_gains_biases = [p for p in model.vit.swinv2.encoder.parameters() if p.ndim < 2]
        nonhidden_params = [*model.vit.swinv2.embeddings.parameters(), *model.vit.classifier.parameters(),*model.vit.swinv2.layernorm.parameters()]
        param_groups = [
            dict(params=hidden_weights, use_muon=True,
                lr=0.00025),
            dict(params=hidden_gains_biases+nonhidden_params, use_muon=False,
                lr=learning_rate, betas=(0.9, 0.95),),
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

           
            # ✅ Wandb log at every step
            wandb.log({"Loss/train_step": loss.item(), "step": global_step})

            global_step += 1  # Increment global step

        if scheduler:
            scheduler.step()
 
        print("Epoch {} training loss:".format(epoch), total_training_loss)
        
        wandb.log({"Loss/train": total_training_loss, "epoch": epoch})
        
        training_perf_dict = dict()
        try:
            training_perf_dict = prec_rec_f1_acc_mcc(all_training_labels, all_training_preds)
        except:
            print("There was a problem during training performance calculation!")
        
        training_roc_auc = roc_auc_score(all_training_labels, np.array(all_training_probs)[:, 1])
        training_pr_auc = average_precision_score(all_training_labels, np.array(all_training_probs)[:, 1])
        training_perf_dict["ROC AUC"] = training_roc_auc
        training_perf_dict["PR AUC"] = training_pr_auc

        for metric, value in training_perf_dict.items():
            wandb.log({f"Train/{metric}": value, "epoch": epoch})


        model.eval()

        # Save the model checkpoint to resume training later
        torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'steps': global_step,
            }, os.path.join(trained_models_path,experiment_name, target_id+"_"+ str_arguments+'-checkpoint.pth'))
        
        with torch.no_grad():
            print("Validation mode:", not model.training)

            total_val_loss, total_val_count, all_val_comp_ids, all_val_labels, val_predictions,val_pred_probs = calculate_val_test_loss(model, criterion, valid_loader, device)

            val_perf_dict = dict()
            val_perf_dict["MCC"] = 0.0
            try:
                val_perf_dict = prec_rec_f1_acc_mcc(all_val_labels, val_predictions)
            except:
                print("There was a problem during validation performance calculation!")


            val_roc_auc = roc_auc_score(all_val_labels, np.array(val_pred_probs)[:, 1])
            val_pr_auc = average_precision_score(all_val_labels, np.array(val_pred_probs)[:, 1])
            val_perf_dict["ROC AUC"] = val_roc_auc
            val_perf_dict["PR AUC"] = val_pr_auc

            for metric, value in val_perf_dict.items():
                wandb.log({f"Validation/{metric}": value, "epoch": epoch})

            total_test_loss, total_test_count, all_test_comp_ids, all_test_labels, test_predictions,test_pred_probs = calculate_val_test_loss(
                model, criterion, test_loader, device)

            test_perf_dict = dict()
            test_perf_dict["MCC"] = 0.0
            try:
                test_perf_dict = prec_rec_f1_acc_mcc(all_test_labels, test_predictions)
            except:
                print("There was a problem during test performance calculation!")


            test_roc_auc = roc_auc_score(all_test_labels, np.array(test_pred_probs)[:, 1])
            test_pr_auc = average_precision_score(all_test_labels, np.array(test_pred_probs)[:, 1])
            test_perf_dict["ROC AUC"] = test_roc_auc
            test_perf_dict["PR AUC"] = test_pr_auc

            for metric, value in test_perf_dict.items():
                wandb.log({f"Test/{metric}": value, "epoch": epoch})

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

    #generate_images(dataset_file, target_id, target_prediction_dataset_path, max_cores)

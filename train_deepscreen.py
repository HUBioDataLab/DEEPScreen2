import os
import random

import torch

import warnings

import numpy as np

import torch.nn as nn
from torch.optim.lr_scheduler import LinearLR

from models import CNNModel1, CNNModel2, ViT, YOLOv11Classifier

from data_loading import get_train_test_val_data_loaders
from evaluation_metrics import (
    binary_ranking_metrics,
    get_list_of_scores,
    prec_rec_f1_acc_mcc,
)

import wandb

import matplotlib.pyplot as plt
import cv2
import shutil
from tqdm import tqdm
from muon import SingleDeviceMuonWithAuxAdam

warnings.filterwarnings(action='ignore')

sep = os.sep

current_path_beginning = os.getcwd().split("DEEPScreen")[0]
current_path_version = os.getcwd().split("DEEPScreen")[1].split("/")[0]

project_file_path = f"{current_path_beginning}DEEPScreen{current_path_version}"
training_files_path = f"{project_file_path}{sep}training_files"
result_files_path = f"{project_file_path}{sep}result_files"
trained_models_path = f"{project_file_path}{sep}trained_models"

def save_best_model_predictions(experiment_name, epoch, validation_scores_dict, test_scores_dict, model, project_file_path, target_id, str_arguments,
                                all_test_comp_ids, test_labels, test_predictions,global_step,optimizer,
                                checkpoint_state=None, save_checkpoint=True):

    if not os.path.exists(os.path.join(trained_models_path, experiment_name)):
            os.makedirs(os.path.join(trained_models_path, experiment_name))
    
    if checkpoint_state is None:
        checkpoint_state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'steps': global_step,
        }
    if save_checkpoint:
        torch.save(
            checkpoint_state,
            os.path.join(
                trained_models_path,
                experiment_name,
                target_id + "_" + str_arguments + '-checkpoint.pth',
            ),
        )

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


def _loader_generator_states(loaders):
    states = []
    for loader in loaders:
        generator = getattr(getattr(loader, "sampler", None), "generator", None)
        states.append(generator.get_state() if generator is not None else None)
    return states


def _restore_loader_generator_states(loaders, states):
    if not states:
        return
    for loader, state in zip(loaders, states):
        generator = getattr(getattr(loader, "sampler", None), "generator", None)
        if generator is not None and state is not None:
            generator.set_state(state)


def _capture_rng_state(loaders):
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "loader_generators": _loader_generator_states(loaders),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state, loaders):
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])
    _restore_loader_generator_states(loaders, state.get("loader_generators"))


def _load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _checkpoint_payload(
    *,
    epoch,
    model,
    optimizer,
    scheduler,
    global_step,
    best_val_score,
    best_test_performance_dict,
    best_test_predictions,
    early_stopping_counter,
    has_best_model,
    config,
    loaders,
):
    return {
        "checkpoint_version": 2,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "steps": global_step,
        "best_val_score": best_val_score,
        "best_test_performance_dict": best_test_performance_dict,
        "best_test_predictions": best_test_predictions,
        "early_stopping_counter": early_stopping_counter,
        "has_best_model": has_best_model,
        "config": dict(config),
        "rng_state": _capture_rng_state(loaders),
    }


def _metric_improved(current_score, best_score, has_best_model):
    if not has_best_model:
        return True
    if np.isnan(current_score):
        return False
    if np.isnan(best_score):
        return True
    return current_score > best_score


def _write_final_results(result_path, prediction_path, performance, predictions):
    with open(result_path, "w", encoding="utf-8") as result_file:
        for score in get_list_of_scores():
            result_file.write(f"Test {score}:\t{performance[score]}\n")
    with open(prediction_path, "w", encoding="utf-8") as prediction_file:
        prediction_file.write(predictions)

def get_device(cuda_selection):
    device = "cpu"
    if torch.cuda.is_available():
        print("GPU is available on this device!")
        device = "cuda:"+str(cuda_selection)
    else:
        print("CPU is available on this device!")
    return device

def calculate_val_test_loss(model, criterion, data_loader, device):
    total_count = 0
    summed_sample_loss = 0.0
    all_comp_ids = []
    all_labels = []
    all_predictions = []
    all_pred_probs = []

    for i, data in enumerate(tqdm(data_loader)):
        img_arrs, labels, comp_ids = data
        img_arrs = img_arrs.to(device=device, dtype=torch.float32)
        labels = labels.to(device=device, dtype=torch.long)
        total_count += len(comp_ids)
        y_pred = model(img_arrs).to(device)
        loss = criterion(y_pred, labels)
        summed_sample_loss += float(loss.item()) * len(comp_ids)
        all_comp_ids.extend(list(comp_ids))
        _, preds = torch.max(y_pred, 1)
        all_labels.extend(list(labels.detach().cpu().numpy()))
        all_predictions.extend(list(preds.detach().cpu().numpy()))
        
        probs = torch.softmax(y_pred, dim=1) 
        all_pred_probs.extend(probs.detach().cpu().numpy())

    mean_loss = summed_sample_loss / total_count if total_count else float("nan")
    return mean_loss, total_count, all_comp_ids, all_labels, all_predictions, all_pred_probs

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
    selection_metric,
    run_seed,

    sweep=False, scheduler=False, use_muon=False, split_seed=None,
    training_data_root=None,
):

    if selection_metric not in ["auroc", "auprc", "mcc"]:
        raise ValueError(f"Invalid selection_metric '{selection_metric}'. Must be one of ['auroc', 'auprc', 'mcc'].")
    if selection_metric == "auroc":
        selection_metric = "ROC AUC"
    elif selection_metric == "auprc":
        selection_metric = "PR AUC"
    elif selection_metric == "mcc":
        selection_metric = "MCC"
    print(f"Using '{selection_metric}' as the metric for model selection during training.")
    # ---- 1. CONFIGURATION MERGE (Fail-safe) ----
    # Add runtime/function args to cfg so they are logged too
    cfg = {
        "target_id": target_id,
        "model_name": model_name,
        "experiment_name": experiment_name,
        "cuda_selection": cuda_selection,
        "scheduler": scheduler,
        "use_muon": use_muon,
        "model_save": model_save,
        "run_seed": run_seed,
        "split_seed": split_seed,
        "training_data_root": str(training_data_root) if training_data_root else None,
    }
    
    for i,v in config.items():
        cfg[i] = v
    # ---- 2. GENERATE RUN STRING ----
    # Create a unique identifier string based on key hyperparameters
    # We select specific keys to be part of the filename/ID
    key_params = [
        target_id, model_name,
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

    # Result paths are written atomically at completion or early stopping.
    res_file_path = os.path.join(exp_path, f"best_val_test_performance_results-{str_arguments}.txt")
    pred_file_path = os.path.join(exp_path, f"best_val_test_predictions-{str_arguments}.txt")

    # Data Loaders
    num_workers = int(cfg.get("num_workers", 4))
    loader_kwargs = {"num_workers": num_workers}
    if training_data_root is not None:
        loader_kwargs["parent_path"] = training_data_root
    train_loader, valid_loader, test_loader = get_train_test_val_data_loaders(
        target_id,
        run_seed,
        cfg['bs'],
        **loader_kwargs,
    )
    loaders = (train_loader, valid_loader, test_loader)

    # ---- 5. DYNAMIC MODEL LOADING ----
    # This is the "Bugless" part. We map model names to classes and specific args.
    model = None
    
    if model_name == "CNNModel1":
        model = CNNModel1(
            cfg['fc1'],
            cfg['fc2'],
            cfg['dropout']
        ).to(device)

    elif model_name == "CNNModel2":
        model = CNNModel2(
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
        model = YOLOv11Classifier(
            num_classes=2,
            model_size=cfg.get("model_size", "yolo11m"),
        ).to(device)
    else:
        raise ValueError(f"Model '{model_name}' is not recognized.")
    # ---- 6. OPTIMIZER, SCHEDULER, CRITERION ----
    n_epoch = int(cfg['epoch'])
    learning_rate = float(cfg['learning_rate'])
    muon_lr = float(cfg.get('muon_lr', 2.5e-4)) if use_muon else None
    end_learning_rate_factor = float(
        cfg.get('end_learning_rate_factor', cfg.get('end_learning_rate', 0.2))
    )
    

    if use_muon: 
        if model_name == "ViT":
            hidden_weights = [p for p in model.vit.swinv2.encoder.parameters() if p.requires_grad and p.ndim == 2]
            hidden_gains_biases = [p for p in model.vit.swinv2.encoder.parameters() if p.requires_grad and p.ndim != 2]
            nonhidden_params = [p for p in [*model.vit.swinv2.embeddings.parameters(), *model.vit.classifier.parameters(),*model.vit.swinv2.layernorm.parameters()] if p.requires_grad]
            param_groups = [
                dict(params=hidden_weights, use_muon=True,
                    lr=muon_lr),
                dict(params=hidden_gains_biases+nonhidden_params, use_muon=False,
                    lr=learning_rate, betas=(0.9, 0.95),),
            ]
            optimizer = SingleDeviceMuonWithAuxAdam(param_groups)

        elif model_name == "YOLOv11":

            seq_model = model.model.model   # Sequential

            backbone_neck = seq_model[:-1]  # 0–9
            head = seq_model[-1]            # Classify layer

            hidden_weights = []
            hidden_gains_biases = []

            for m in backbone_neck:
                for p in m.parameters():
                    if not p.requires_grad:
                        continue
                    # muon-optimizer 0.1.0 flattens convolutional updates but
                    # does not reshape them before p.add_; keep 4-D kernels on
                    # the auxiliary Adam path.
                    if p.ndim == 2:
                        hidden_weights.append(p)
                    else:
                        hidden_gains_biases.append(p)

            head_params = [p for p in head.parameters() if p.requires_grad]

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

            hidden_weights = [p for p in model.parameters() if p.requires_grad and p.ndim == 2]
            
            hidden_gains_biases = [p for p in model.parameters() if p.requires_grad and p.ndim != 2]

            param_groups = [
                dict(params=hidden_weights, use_muon=True, 
                     lr=muon_lr), 
                
                dict(params=hidden_gains_biases, use_muon=False, 
                     lr=learning_rate, betas=(0.9, 0.95)),
            ]
            optimizer = SingleDeviceMuonWithAuxAdam(param_groups)
    else:
        optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=learning_rate,
        )

    checkpoint = None
    start_epoch = 0
    start_step = 0
    if model_save != "None":
        checkpoint = _load_checkpoint(model_save, device)
        if "model_state_dict" not in checkpoint:
            # Preserve support for legacy raw state-dict files. They initialize
            # weights but do not contain enough state to resume an epoch.
            model.load_state_dict(checkpoint)
            checkpoint = None
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = int(checkpoint['epoch']) + 1
            start_step = int(checkpoint.get('steps', 0))

    scheduler_instance = None
    if scheduler:
        scheduler_instance = LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=end_learning_rate_factor,
            total_iters=max(n_epoch, 1),
        )
        if checkpoint and checkpoint.get("scheduler_state_dict") is not None:
            scheduler_instance.load_state_dict(checkpoint["scheduler_state_dict"])
            

    criterion = nn.CrossEntropyLoss()
    optimizer.zero_grad()

    best_val_score = checkpoint.get("best_val_score", -float("inf")) if checkpoint else -float("inf")

    best_test_performance_dict = checkpoint.get("best_test_performance_dict", {}) if checkpoint else {}
    best_test_predictions = checkpoint.get("best_test_predictions", "") if checkpoint else ""
    has_best_model = checkpoint.get("has_best_model", bool(best_test_performance_dict)) if checkpoint else False

    early_stopping_counter = checkpoint.get("early_stopping_counter", 0) if checkpoint else 0

    global_step = start_step  # Track steps across all epochs

    if checkpoint:
        _restore_rng_state(checkpoint.get("rng_state"), loaders)

    checkpoint_path = os.path.join(
        trained_models_path,
        experiment_name,
        target_id + "_" + str_arguments + '-checkpoint.pth',
    )

    if start_epoch >= n_epoch:
        print(
            f"Checkpoint already completed epoch {start_epoch - 1}; "
            f"configured total epochs: {n_epoch}."
        )
        if has_best_model:
            _write_final_results(
                res_file_path,
                pred_file_path,
                best_test_performance_dict,
                best_test_predictions,
            )
        wandb.finish()
        return

    for epoch in range(start_epoch, n_epoch):
        total_training_count = 0
        summed_training_sample_loss = 0.0
        print(f"Epoch :{epoch}")
        model.train()
        all_training_labels = []
        all_training_preds = []
        all_training_probs = []
        print("Training mode:", model.training)

        for data in tqdm(train_loader):
            optimizer.zero_grad()
            img_arrs, labels, comp_ids = data
            img_arrs = img_arrs.to(device=device, dtype=torch.float32)
            labels = labels.to(device=device, dtype=torch.long)
            total_training_count += len(comp_ids)

            y_pred = model(img_arrs)
            preds = torch.argmax(y_pred, dim=1)
            probabilities = torch.softmax(y_pred, dim=1)
            all_training_labels.extend(labels.detach().cpu().tolist())
            all_training_preds.extend(preds.detach().cpu().tolist())
            all_training_probs.extend(probabilities.detach().cpu().numpy())

            loss = criterion(y_pred, labels)
            summed_training_sample_loss += float(loss.item()) * len(comp_ids)
            loss.backward()
            optimizer.step()

            wandb.log({"Loss/train_step": loss.item(), "step": global_step})
            global_step += 1

        if scheduler_instance is not None:
            scheduler_instance.step()

        mean_training_loss = (
            summed_training_sample_loss / total_training_count
            if total_training_count
            else float("nan")
        )
        print(f"Epoch {epoch} mean training loss:", mean_training_loss)
        wandb.log({"Loss/train": mean_training_loss, "epoch": epoch})

        training_perf_dict = prec_rec_f1_acc_mcc(
            all_training_labels,
            all_training_preds,
        )
        training_perf_dict.update(
            binary_ranking_metrics(
                all_training_labels,
                [probability[1] for probability in all_training_probs],
            )
        )
        for metric, value in training_perf_dict.items():
            wandb.log({f"Train/{metric}": value, "epoch": epoch})

        model.eval()
        with torch.no_grad():
            print("Validation mode:", not model.training)
            val_loss, _, raw_val_comp_ids, raw_val_labels, raw_val_predictions, raw_val_probs = calculate_val_test_loss(
                model, criterion, valid_loader, device
            )
            _, all_val_labels, val_predictions, val_pred_probs = aggregate_predictions(
                raw_val_comp_ids,
                raw_val_labels,
                raw_val_predictions,
                raw_val_probs,
            )
            val_perf_dict = prec_rec_f1_acc_mcc(all_val_labels, val_predictions)
            val_perf_dict.update(
                binary_ranking_metrics(
                    all_val_labels,
                    [probability[1] for probability in val_pred_probs],
                )
            )
            for metric, value in val_perf_dict.items():
                wandb.log({f"Validation/{metric}": value, "epoch": epoch})

            # Kept intentionally for backward-compatible reporting behavior.
            test_loss, _, raw_test_comp_ids, raw_test_labels, raw_test_predictions, raw_test_probs = calculate_val_test_loss(
                model, criterion, test_loader, device
            )
            all_test_comp_ids, all_test_labels, test_predictions, test_pred_probs = aggregate_predictions(
                raw_test_comp_ids,
                raw_test_labels,
                raw_test_predictions,
                raw_test_probs,
            )
            test_perf_dict = prec_rec_f1_acc_mcc(all_test_labels, test_predictions)
            test_perf_dict.update(
                binary_ranking_metrics(
                    all_test_labels,
                    [probability[1] for probability in test_pred_probs],
                )
            )
            for metric, value in test_perf_dict.items():
                wandb.log({f"Test/{metric}": value, "epoch": epoch})

        current_val_score = float(val_perf_dict[selection_metric])
        improved = _metric_improved(
            current_val_score,
            float(best_val_score),
            has_best_model,
        )

        if early_stopping and epoch >= warmup:
            print(f"Val {selection_metric} score: ", current_val_score)
            print(f"Best val {selection_metric}: ", best_val_score)
            early_stopping_counter = 0 if improved else early_stopping_counter + 1
            print("Early stopping number : ", early_stopping_counter)

        if improved:
            best_val_score = current_val_score
            has_best_model = True
            _, best_test_performance_dict, best_test_predictions, _ = save_best_model_predictions(
                experiment_name,
                epoch,
                val_perf_dict,
                test_perf_dict,
                model,
                project_file_path,
                target_id,
                str_arguments,
                all_test_comp_ids,
                all_test_labels,
                test_predictions,
                global_step,
                optimizer,
                save_checkpoint=False,
            )

        wandb.log({"Loss/validation": val_loss, "epoch": epoch})
        wandb.log({"Loss/test": test_loss, "epoch": epoch})

        torch.save(
            _checkpoint_payload(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler_instance,
                global_step=global_step,
                best_val_score=best_val_score,
                best_test_performance_dict=best_test_performance_dict,
                best_test_predictions=best_test_predictions,
                early_stopping_counter=early_stopping_counter,
                has_best_model=has_best_model,
                config=cfg,
                loaders=loaders,
            ),
            checkpoint_path,
        )

        if early_stopping and epoch >= warmup and early_stopping_counter >= patience:
            _write_final_results(
                res_file_path,
                pred_file_path,
                best_test_performance_dict,
                best_test_predictions,
            )
            print(f"Early stopping triggered at epoch {epoch}")
            wandb.finish()
            return

    _write_final_results(
        res_file_path,
        pred_file_path,
        best_test_performance_dict,
        best_test_predictions,
    )
    wandb.finish()

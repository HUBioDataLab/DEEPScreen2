import numpy as np
import copy
from math import sqrt
from scipy import stats
from sklearn import preprocessing, metrics
from sklearn.metrics import confusion_matrix


def _to_flat_numpy(values):
    flat_values = []
    for value in values:
        if hasattr(value, "item"):
            flat_values.append(value.item())
        else:
            flat_values.append(value)
    return np.asarray(flat_values, dtype=float)


def prec_rec_f1_acc_mcc(y_true, y_pred):
    performance_threshold_dict  = dict()

    y_true_tmp = []
    for each_y_true in y_true:
        y_true_tmp.append(each_y_true.item())
    y_true = y_true_tmp

    y_pred_tmp = []
    for each_y_pred in y_pred:
        y_pred_tmp.append(each_y_pred.item())
    y_pred = y_pred_tmp


    precision = metrics.precision_score(y_true, y_pred)
    recall = metrics.recall_score(y_true, y_pred)
    f1_score = metrics.f1_score(y_true, y_pred)
    accuracy = metrics.accuracy_score(y_true, y_pred)
    mcc = metrics.matthews_corrcoef(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    performance_threshold_dict["Precision"] = precision
    performance_threshold_dict["Recall"] = recall
    performance_threshold_dict["F1-Score"] = f1_score
    performance_threshold_dict["Accuracy"] = accuracy
    performance_threshold_dict["MCC"] = mcc
    performance_threshold_dict["TP"] = tp
    performance_threshold_dict["FP"] = fp
    performance_threshold_dict["TN"] = tn
    performance_threshold_dict["FN"] = fn


    return performance_threshold_dict


def regression_metrics(y_true, y_pred):
    y_true = _to_flat_numpy(y_true)
    y_pred = _to_flat_numpy(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
    if y_true.size == 0:
        raise ValueError("y_true and y_pred must contain at least one value")

    abs_errors = np.abs(y_true - y_pred)
    mae = float(np.mean(abs_errors))

    true_mean = float(np.mean(y_true))
    baseline_abs_error = float(np.sum(np.abs(y_true - true_mean)))
    model_abs_error = float(np.sum(abs_errors))
    rae = float(model_abs_error / baseline_abs_error) if baseline_abs_error != 0 else 0.0

    r2 = float(metrics.r2_score(y_true, y_pred))
    spearman_rho = float(stats.spearmanr(y_true, y_pred).statistic)
    kendall_tau = float(stats.kendalltau(y_true, y_pred).statistic)

    return {
        "MAE": mae,
        "RAE": rae,
        "R2": r2,
        "Spearman_R": spearman_rho,
        "Kendall's_Tau": kendall_tau,
    }


def get_list_of_scores():
    return ["Precision", "Recall", "F1-Score", "Accuracy", "MCC", "TP", "FP", "TN", "FN"]


def get_list_of_regression_scores():
    return ["MAE", "RAE", "R2", "Spearman_R", "Kendall's_Tau"]

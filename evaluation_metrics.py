import numpy as np
import copy
from math import sqrt
from scipy import stats
from sklearn import preprocessing, metrics
from sklearn.metrics import confusion_matrix


def _to_python_list(values):
    return [value.item() if hasattr(value, "item") else value for value in values]


def prec_rec_f1_acc_mcc(y_true, y_pred):
    performance_threshold_dict = dict()
    y_true = _to_python_list(y_true)
    y_pred = _to_python_list(y_pred)

    precision = metrics.precision_score(y_true, y_pred, zero_division=0)
    recall = metrics.recall_score(y_true, y_pred, zero_division=0)
    f1_score = metrics.f1_score(y_true, y_pred, zero_division=0)
    accuracy = metrics.accuracy_score(y_true, y_pred)
    mcc = (
        metrics.matthews_corrcoef(y_true, y_pred)
        if len(set(y_true)) > 1 and len(set(y_pred)) > 1
        else 0.0
    )
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    performance_threshold_dict["Precision"] = float(precision)
    performance_threshold_dict["Recall"] = float(recall)
    performance_threshold_dict["F1-Score"] = float(f1_score)
    performance_threshold_dict["Accuracy"] = float(accuracy)
    performance_threshold_dict["MCC"] = float(mcc)
    performance_threshold_dict["TP"] = int(tp)
    performance_threshold_dict["FP"] = int(fp)
    performance_threshold_dict["TN"] = int(tn)
    performance_threshold_dict["FN"] = int(fn)

    return performance_threshold_dict


def binary_ranking_metrics(y_true, positive_probabilities):
    """Compute ranking metrics without crashing on a one-class split."""
    y_true = _to_python_list(y_true)
    positive_probabilities = _to_python_list(positive_probabilities)
    if len(set(y_true)) < 2:
        return {"ROC AUC": float("nan"), "PR AUC": float("nan")}
    return {
        "ROC AUC": float(metrics.roc_auc_score(y_true, positive_probabilities)),
        "PR AUC": float(metrics.average_precision_score(y_true, positive_probabilities)),
    }

def get_list_of_scores():
    return ["Precision", "Recall", "F1-Score", "Accuracy", "MCC", "TP", "FP", "TN", "FN"]

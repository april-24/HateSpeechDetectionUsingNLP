"""Evaluation helpers.

Primary harmful-content YES/NO metrics are reported separately from the
six-label multilabel metrics. The ``abusive`` label is the only label allowed
to determine the final harmful-content verdict.
"""
import os
import numpy as np
import pandas as pd
from .config import RESULTS_DIR

from sklearn.metrics import (
    accuracy_score, hamming_loss, precision_recall_fscore_support,
    classification_report, confusion_matrix
)

RESULTS_CSV = str(RESULTS_DIR / "model_scores.csv")


def primary_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]).ravel()
    precision = (tp / max(tp + fp, 1))
    recall = (tp / max(tp + fn, 1))
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    fpr = fp / max(fp + tn, 1)
    fnr = fn / max(fn + tp, 1)
    return {
        "harmful_precision": round(float(precision), 4),
        "harmful_recall": round(float(recall), 4),
        "harmful_f1": round(float(f1), 4),
        "harmful_false_positive_rate": round(float(fpr), 4),
        "harmful_false_negative_rate": round(float(fnr), 4),
        "harmful_tn": int(tn), "harmful_fp": int(fp),
        "harmful_fn": int(fn), "harmful_tp": int(tp),
    }


def evaluate_model(name, y_true, y_pred, label_names,
                   train_time=None, predict_time=None,
                   primary_probabilities=None, primary_threshold=None):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    subset_acc = accuracy_score(y_true, y_pred)
    hloss = hamming_loss(y_true, y_pred)
    label_acc = 1 - hloss

    per_label_acc = {
        label_names[i]: accuracy_score(y_true[:, i], y_pred[:, i])
        for i in range(len(label_names))
    }
    p_mi, r_mi, f_mi, _ = precision_recall_fscore_support(
        y_true, y_pred, average="micro", zero_division=0)
    p_ma, r_ma, f_ma, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0)
    p_we, r_we, f_we, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0)

    primary = primary_metrics(y_true[:, 0], y_pred[:, 0])

    print(f"\n{'='*70}\n  RESULTS: {name}\n{'='*70}")
    print("PRIMARY HARMFUL-CONTENT VERDICT (abusive only)")
    print(f"Precision : {primary['harmful_precision']:.4f}")
    print(f"Recall    : {primary['harmful_recall']:.4f}")
    print(f"F1        : {primary['harmful_f1']:.4f}")
    print(f"FPR       : {primary['harmful_false_positive_rate']:.4f}")
    print(f"FNR       : {primary['harmful_false_negative_rate']:.4f}")
    print(f"Confusion: TN={primary['harmful_tn']} FP={primary['harmful_fp']} "
          f"FN={primary['harmful_fn']} TP={primary['harmful_tp']}")
    if primary_threshold is not None:
        print(f"Abusive threshold: {primary_threshold:.2f}")

    print("\nMULTILABEL METRICS (six labels)")
    print(f"Per-label accuracy average : {label_acc:.4f}")
    print(f"Subset accuracy             : {subset_acc:.4f}")
    print(f"Hamming loss                : {hloss:.4f}")
    print(f"Micro    -> P: {p_mi:.4f} R: {r_mi:.4f} F1: {f_mi:.4f}")
    print(f"Macro    -> P: {p_ma:.4f} R: {r_ma:.4f} F1: {f_ma:.4f}")
    print(f"Weighted -> P: {p_we:.4f} R: {r_we:.4f} F1: {f_we:.4f}")
    print("\nPer-label report:")
    print(classification_report(
        y_true, y_pred, target_names=label_names, zero_division=0))

    result = {
        "model": name,
        **primary,
        "accuracy": round(float(label_acc), 4),
        "subset_accuracy": round(float(subset_acc), 4),
        "hamming_loss": round(float(hloss), 4),
        "precision_micro": round(float(p_mi), 4),
        "recall_micro": round(float(r_mi), 4),
        "f1_micro": round(float(f_mi), 4),
        "precision_macro": round(float(p_ma), 4),
        "recall_macro": round(float(r_ma), 4),
        "f1_macro": round(float(f_ma), 4),
        "precision_weighted": round(float(p_we), 4),
        "recall_weighted": round(float(r_we), 4),
        "f1_weighted": round(float(f_we), 4),
    }
    if train_time is not None:
        result["train_time_sec"] = round(float(train_time), 3)
    if predict_time is not None:
        result["predict_time_sec"] = round(float(predict_time), 4)
    if primary_threshold is not None:
        result["threshold_abusive"] = float(primary_threshold)
    result["_per_label_accuracy"] = per_label_acc
    return result


def save_result(result: dict, path: str = RESULTS_CSV):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    row = {k: v for k, v in result.items() if not k.startswith("_")}
    if os.path.exists(path):
        df = pd.read_csv(path)
        df = df[df["model"] != row["model"]]
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(path, index=False)
    print(f"Saved scores for '{row['model']}' -> {path}")

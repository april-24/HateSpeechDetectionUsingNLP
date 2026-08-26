"""Shared evaluation helpers for primary harmful detection and multilabel outputs."""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    hamming_loss,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

RESULTS_CSV = os.path.join("results", "model_scores.csv")


def _binary_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) else 0.0
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def evaluate_model(name, y_true, y_pred, label_names,
                   train_time=None, predict_time=None,
                   probabilities=None, thresholds=None):
    """Evaluate primary harmful-content detection separately from multilabel metrics."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    primary_idx = list(label_names).index("abusive")
    primary = _binary_metrics(y_true[:, primary_idx], y_pred[:, primary_idx])

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

    print(f"\n{'='*68}\n  RESULTS: {name}\n{'='*68}")
    print("PRIMARY harmful-content verdict (abusive)")
    print(f"  Accuracy : {primary['accuracy']:.4f}")
    print(f"  Precision: {primary['precision']:.4f}")
    print(f"  Recall   : {primary['recall']:.4f}")
    print(f"  F1       : {primary['f1']:.4f}")
    print(f"  FPR      : {primary['false_positive_rate']:.4f}")
    print(f"  FNR      : {primary['false_negative_rate']:.4f}")
    print(f"  Confusion: TN={primary['tn']} FP={primary['fp']} FN={primary['fn']} TP={primary['tp']}")
    print("\nMULTILABEL")
    print(f"  Accuracy (per-label avg): {label_acc:.4f}")
    print(f"  Subset accuracy         : {subset_acc:.4f}")
    print(f"  Hamming loss            : {hloss:.4f}")
    print(f"  Micro    -> P: {p_mi:.4f}  R: {r_mi:.4f}  F1: {f_mi:.4f}")
    print(f"  Macro    -> P: {p_ma:.4f}  R: {r_ma:.4f}  F1: {f_ma:.4f}")
    print(f"  Weighted -> P: {p_we:.4f}  R: {r_we:.4f}  F1: {f_we:.4f}")
    if train_time is not None:
        print(f"Training time              : {train_time:.2f}s")
    if predict_time is not None:
        print(f"Prediction time (test set): {predict_time:.4f}s")
    print("\nPer-label report:")
    print(classification_report(y_true, y_pred, target_names=label_names,
                                zero_division=0))

    result = {
        "model": name,
        "primary_accuracy": round(primary["accuracy"], 4),
        "primary_precision": round(primary["precision"], 4),
        "primary_recall": round(primary["recall"], 4),
        "primary_f1": round(primary["f1"], 4),
        "primary_false_positive_rate": round(primary["false_positive_rate"], 4),
        "primary_false_negative_rate": round(primary["false_negative_rate"], 4),
        "primary_tn": primary["tn"], "primary_fp": primary["fp"],
        "primary_fn": primary["fn"], "primary_tp": primary["tp"],
        "accuracy": round(label_acc, 4),
        "subset_accuracy": round(subset_acc, 4),
        "hamming_loss": round(hloss, 4),
        "precision_micro": round(p_mi, 4),
        "recall_micro": round(r_mi, 4),
        "f1_micro": round(f_mi, 4),
        "precision_macro": round(p_ma, 4),
        "recall_macro": round(r_ma, 4),
        "f1_macro": round(f_ma, 4),
        "precision_weighted": round(p_we, 4),
        "recall_weighted": round(r_we, 4),
        "f1_weighted": round(f_we, 4),
    }
    if train_time is not None:
        result["train_time_sec"] = round(train_time, 3)
    if predict_time is not None:
        result["predict_time_sec"] = round(predict_time, 4)
    if thresholds:
        result["primary_threshold"] = float(thresholds.get("abusive", 0.5))

    result["_per_label_accuracy"] = per_label_acc
    return result


def save_result(result: dict, path: str = RESULTS_CSV):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    row = {k: v for k, v in result.items() if not k.startswith("_")}
    if os.path.exists(path):
        df = pd.read_csv(path)
        if "model" in df.columns:
            df = df[df["model"] != row["model"]]
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(path, index=False)
    print(f"\nSaved scores for '{row['model']}' -> {path}")

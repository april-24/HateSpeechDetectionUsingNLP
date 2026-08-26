"""
evaluate.py
-----------
Shared evaluation helpers so every member reports metrics the SAME way.

For multi-label classification there are TWO different things people call
"accuracy" - and mixing them up makes results look worse than they are:

    - Per-label accuracy (a.k.a. Hamming accuracy / label-based accuracy):
        each of the 6 categories is scored as its own yes/no classification
        question, then averaged. THIS is the number that lines up with
        "accuracy" as reported in most multi-label classification papers.
        Equivalent to (1 - Hamming loss).

    - Subset accuracy (exact match ratio): the strict, all-or-nothing
        version - a row only counts as correct if ALL 6 labels are right
        simultaneously. This is a much harder bar and looks low even for a
        genuinely good model - that's expected behaviour for multi-label
        problems, not a sign the model is bad. Kept here for transparency,
        clearly labeled as the stricter measure.

Also reported: Precision / Recall / F1 with 'micro' and 'macro' averaging
    micro = aggregate over all label decisions (favours frequent labels)
    macro = average of per-label scores (treats every label equally)

`evaluate_model` returns a dict of the headline numbers and prints a full
report. `save_result` appends a model's headline numbers to a shared CSV so
compare_models.py / the app can build the final comparison table.
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    hamming_loss,
    precision_recall_fscore_support,
    classification_report,
)

RESULTS_CSV = os.path.join("results", "model_scores.csv")


def evaluate_model(name, y_true, y_pred, label_names,
                   train_time=None, predict_time=None):
    """Print a full multi-label report and return headline metrics as a dict.

    train_time / predict_time (seconds, optional): pass these in from the
    training script if you want Training Time / Prediction Time reported
    alongside the accuracy metrics (common in benchmark-style comparisons).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    subset_acc = accuracy_score(y_true, y_pred)
    hloss = hamming_loss(y_true, y_pred)
    label_acc = 1 - hloss  # == mean of per-label accuracy_score across labels

    # Per-label accuracy, individually, for a category-by-category table
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

    print(f"\n{'='*60}\n  RESULTS: {name}\n{'='*60}")
    print(f"Accuracy (per-label, avg)   : {label_acc:.4f}   <- comparable to typical papers")
    print(f"Subset accuracy (all 6 exact): {subset_acc:.4f}   <- stricter, expected to look lower")
    print(f"Hamming loss (lower better)  : {hloss:.4f}")
    print(f"Micro    -> P: {p_mi:.4f}  R: {r_mi:.4f}  F1: {f_mi:.4f}")
    print(f"Macro    -> P: {p_ma:.4f}  R: {r_ma:.4f}  F1: {f_ma:.4f}")
    print(f"Weighted -> P: {p_we:.4f}  R: {r_we:.4f}  F1: {f_we:.4f}")
    if train_time is not None:
        print(f"Training time                : {train_time:.2f}s")
    if predict_time is not None:
        print(f"Prediction time (test set)   : {predict_time:.4f}s")
    print("\nPer-label report:")
    print(classification_report(y_true, y_pred, target_names=label_names,
                                zero_division=0))

    result = {
        "model": name,
        "accuracy": round(label_acc, 4),          # headline, paper-comparable
        "subset_accuracy": round(subset_acc, 4),  # strict, kept for transparency
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

    # stash per-label accuracy too (not written to the summary CSV row, but
    # returned so the app / a report script can show a per-category table)
    result["_per_label_accuracy"] = per_label_acc
    return result


def save_result(result: dict, path: str = RESULTS_CSV):
    """Append (or replace) one model's scores in the shared results CSV.
    The nested per-label accuracy breakdown (if present) is excluded from the
    flat CSV row - the app recomputes it live on the Model Evaluation page."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    row = {k: v for k, v in result.items() if not k.startswith("_")}
    if os.path.exists(path):
        df = pd.read_csv(path)
        df = df[df["model"] != row["model"]]   # replace old entry if re-run
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(path, index=False)
    print(f"\nSaved scores for '{row['model']}' -> {path}")

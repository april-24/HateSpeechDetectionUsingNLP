"""Leakage-safe training shared by all three member solutions.

Workflow:
1. Reserve a stratified 20% final test set.
2. Produce out-of-fold predicted probabilities with five folds on the 80% development set.
3. Select a dedicated threshold for every label from pooled OOF predictions.
4. Select the model using the primary ``abusive`` label's OOF F1.
5. Fit the final pipeline on the complete development set.
6. Evaluate exactly once on the untouched final test set.

The deployed application uses a two-stage decision:
Stage 1 -> primary abusive/harmful-content decision.
Stage 2 -> target-group labels are reported only when Stage 1 is positive.
"""

import os
import re
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import FeatureUnion

from .common import prepare_data, stratification_key, RANDOM_STATE
from .evaluate import evaluate_model, save_result
from .config import PRIMARY_LABEL


def build_word_char_features(word_max_features=30000, char_max_features=10000,
                             word_ngram_range=(1, 2)):
    return FeatureUnion([
        ("word", TfidfVectorizer(max_features=word_max_features,
                                 ngram_range=word_ngram_range,
                                 min_df=2, sublinear_tf=True)),
        ("char", TfidfVectorizer(max_features=char_max_features,
                                 analyzer="char_wb", ngram_range=(3, 5),
                                 min_df=2, sublinear_tf=True)),
    ])


def label_probabilities(pipeline, texts):
    """Return model probabilities without fabricating probabilities from margins."""
    if not hasattr(pipeline, "predict_proba"):
        raise TypeError(
            "This pipeline has no predict_proba. Calibrate margin-based models "
            "inside the training pipeline before using probability thresholds."
        )
    values = np.asarray(pipeline.predict_proba(texts))
    return values.reshape(len(texts), -1)


def choose_label_thresholds(y_true, probabilities, labels,
                            lo=0.20, hi=0.80, step=0.01):
    """Select one F1-optimal threshold independently for each label."""
    thresholds = {}
    label_f1 = {}
    for idx, label in enumerate(labels):
        best_threshold, best_f1 = 0.50, -1.0
        y_col = np.asarray(y_true)[:, idx]
        p_col = np.asarray(probabilities)[:, idx]
        for threshold in np.arange(lo, hi + step / 2, step):
            score = f1_score(y_col, p_col >= threshold, zero_division=0)
            if score > best_f1:
                best_threshold, best_f1 = float(threshold), float(score)
        thresholds[label] = round(best_threshold, 2)
        label_f1[label] = round(float(best_f1), 4)
    return thresholds, label_f1


def apply_two_stage_predictions(probabilities, labels, thresholds):
    """Apply label-specific thresholds, then gate target labels on the primary label."""
    probabilities = np.asarray(probabilities)
    raw = np.zeros_like(probabilities, dtype=int)
    for idx, label in enumerate(labels):
        raw[:, idx] = (probabilities[:, idx] >= float(thresholds[label])).astype(int)

    primary_idx = labels.index(PRIMARY_LABEL)
    final = raw.copy()
    target_indices = [i for i, label in enumerate(labels) if label != PRIMARY_LABEL]
    if target_indices:
        final[:, target_indices] = final[:, target_indices] * raw[:, [primary_idx]]
    return final, raw


def _save_per_label(model_name, y_true, y_pred, labels):
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0)
    out = pd.DataFrame({
        "label": labels,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
    })
    slug = re.sub(r"[^a-z0-9]+", "_", model_name.lower()).strip("_")
    out.to_csv(os.path.join("results", f"per_label_{slug}.csv"), index=False)


def train_and_save(model_name, pipeline, model_path, sample=None):
    X_dev, X_test, y_dev, y_test, labels = prepare_data(sample=sample)
    os.makedirs("results", exist_ok=True)

    fold_key = stratification_key(y_dev, min_count=5)
    splitter = StratifiedKFold(n_splits=5, shuffle=True,
                               random_state=RANDOM_STATE)
    oof_probabilities = np.zeros((len(X_dev), len(labels)), dtype=float)
    fold_indices = []

    print(f"\nFive-fold out-of-fold threshold selection: {model_name}")
    cv_started = time.time()
    for fold, (train_idx, valid_idx) in enumerate(
            splitter.split(X_dev, fold_key), start=1):
        fold_pipeline = clone(pipeline)
        fold_pipeline.fit(X_dev.iloc[train_idx], y_dev.iloc[train_idx])
        oof_probabilities[valid_idx] = label_probabilities(
            fold_pipeline, list(X_dev.iloc[valid_idx]))
        fold_indices.append(valid_idx)
        print(f"  completed fold {fold}/5")

    label_thresholds, oof_label_f1 = choose_label_thresholds(
        y_dev.values, oof_probabilities, labels)
    primary_idx = labels.index(PRIMARY_LABEL)
    primary_oof_f1 = float(oof_label_f1[PRIMARY_LABEL])
    primary_threshold = float(label_thresholds[PRIMARY_LABEL])

    fold_primary_f1 = []
    for idx in fold_indices:
        fold_pred = (oof_probabilities[idx, primary_idx] >= primary_threshold).astype(int)
        fold_primary_f1.append(
            f1_score(y_dev.iloc[idx, primary_idx].values,
                     fold_pred, zero_division=0)
        )

    cv_time = time.time() - cv_started
    print(f"Selected label thresholds: {label_thresholds}")
    print(f"OOF primary ({PRIMARY_LABEL}) F1: {primary_oof_f1:.4f}; "
          f"fold mean={np.mean(fold_primary_f1):.4f}, "
          f"std={np.std(fold_primary_f1, ddof=1):.4f}")

    print(f"\nTraining final {model_name} on the complete 80% development set ...")
    started = time.time()
    pipeline.fit(X_dev, y_dev)
    train_time = time.time() - started

    started = time.time()
    test_probabilities = label_probabilities(pipeline, list(X_test))
    y_pred, y_pred_raw = apply_two_stage_predictions(
        test_probabilities, labels, label_thresholds)
    predict_time = time.time() - started

    result = evaluate_model(model_name, y_test.values, y_pred, labels,
                            train_time=train_time, predict_time=predict_time)
    primary_test_p, primary_test_r, primary_test_f1, _ = precision_recall_fscore_support(
        y_test.values[:, primary_idx], y_pred[:, primary_idx],
        average="binary", zero_division=0)
    result.update({
        "primary_label": PRIMARY_LABEL,
        "primary_threshold": primary_threshold,
        "primary_oof_f1": round(primary_oof_f1, 4),
        "primary_oof_f1_mean": round(float(np.mean(fold_primary_f1)), 4),
        "primary_oof_f1_std": round(float(np.std(fold_primary_f1, ddof=1)), 4),
        "primary_test_precision": round(float(primary_test_p), 4),
        "primary_test_recall": round(float(primary_test_r), 4),
        "primary_test_f1": round(float(primary_test_f1), 4),
        "label_thresholds": "|".join(f"{k}={v:.2f}" for k, v in label_thresholds.items()),
        "cv_selection_time_sec": round(cv_time, 3),
    })
    save_result(result)
    _save_per_label(model_name, y_test.values, y_pred, labels)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump({
        "pipeline": pipeline,
        "labels": labels,
        "threshold": primary_threshold,  # backwards-compatible primary threshold
        "primary_threshold": primary_threshold,
        "label_thresholds": label_thresholds,
        "threshold_source": "5-fold pooled out-of-fold predictions on development set, independently per label",
        "selection_metric": "primary abusive F1",
        "primary_oof_f1": primary_oof_f1,
        "primary_test_metrics": {
            "precision": float(primary_test_p),
            "recall": float(primary_test_r),
            "f1": float(primary_test_f1),
        },
        "cv_fold_primary_f1": fold_primary_f1,
    }, model_path, compress=3)
    print(f"Saved trained model -> {model_path}")
    return pipeline

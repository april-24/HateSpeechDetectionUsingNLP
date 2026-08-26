"""Leakage-safe training shared by all three member solutions.

Workflow:
1. Reserve a stratified 20% final test set.
2. Produce out-of-fold probabilities with five folds on the 80% development set.
3. Select one global threshold per model from pooled out-of-fold predictions.
4. Fit the final pipeline on the complete development set.
5. Evaluate once on the untouched final test set and save the complete bundle.
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


def choose_threshold(y_true, probabilities, lo=0.20, hi=0.80, step=0.01):
    """Select the pooled out-of-fold micro-F1-optimal global threshold."""
    best_threshold, best_f1 = 0.50, -1.0
    for threshold in np.arange(lo, hi + step / 2, step):
        score = f1_score(y_true, probabilities >= threshold,
                         average="micro", zero_division=0)
        if score > best_f1:
            best_threshold, best_f1 = float(threshold), float(score)
    return round(best_threshold, 2), best_f1


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

    threshold, oof_micro_f1 = choose_threshold(
        y_dev.values, oof_probabilities)
    fold_f1 = [
        f1_score(y_dev.iloc[idx].values,
                 oof_probabilities[idx] >= threshold,
                 average="micro", zero_division=0)
        for idx in fold_indices
    ]
    cv_time = time.time() - cv_started
    print(f"Selected threshold from pooled OOF predictions: {threshold:.2f}")
    print(f"OOF micro-F1: {oof_micro_f1:.4f}; "
          f"fold mean={np.mean(fold_f1):.4f}, std={np.std(fold_f1, ddof=1):.4f}")

    print(f"\nTraining final {model_name} on the complete 80% development set ...")
    started = time.time()
    pipeline.fit(X_dev, y_dev)
    train_time = time.time() - started

    started = time.time()
    test_probabilities = label_probabilities(pipeline, list(X_test))
    y_pred = (test_probabilities >= threshold).astype(int)
    predict_time = time.time() - started

    result = evaluate_model(model_name, y_test.values, y_pred, labels,
                            train_time=train_time, predict_time=predict_time)
    result.update({
        "threshold": threshold,
        "cv_f1_micro_mean": round(float(np.mean(fold_f1)), 4),
        "cv_f1_micro_std": round(float(np.std(fold_f1, ddof=1)), 4),
        "oof_f1_micro": round(oof_micro_f1, 4),
        "cv_selection_time_sec": round(cv_time, 3),
    })
    save_result(result)
    _save_per_label(model_name, y_test.values, y_pred, labels)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump({
        "pipeline": pipeline,
        "labels": labels,
        "threshold": threshold,
        "threshold_source": "5-fold pooled out-of-fold predictions on development set",
        "cv_fold_f1_micro": fold_f1,
    }, model_path, compress=3)
    print(f"Saved trained model -> {model_path}")
    return pipeline

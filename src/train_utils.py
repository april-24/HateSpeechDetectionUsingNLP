"""Leakage-safe training utilities for the HateXplain multi-label project.

The final 20% test set is NEVER used for model/threshold selection.  All
thresholds are selected from pooled out-of-fold (OOF) predictions produced on
the 80% development set, one threshold per label.
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
from .config import PRIMARY_LABEL, PRIMARY_THRESHOLD_METRIC, PRIMARY_F_BETA
from .evaluate import evaluate_model, save_result


def build_word_char_features(word_max_features=30000, char_max_features=10000,
                             word_ngram_range=(1, 2), char_ngram_range=(3, 5),
                             word_min_df=2, char_min_df=2):
    """Build the common leakage-safe word + character TF-IDF feature union."""
    return FeatureUnion([
        ("word", TfidfVectorizer(
            max_features=word_max_features,
            ngram_range=word_ngram_range,
            min_df=word_min_df,
            sublinear_tf=True)),
        ("char", TfidfVectorizer(
            max_features=char_max_features,
            analyzer="char_wb",
            ngram_range=char_ngram_range,
            min_df=char_min_df,
            sublinear_tf=True)),
    ])


def label_probabilities(pipeline, texts):
    """Return probability estimates from a fitted pipeline."""
    if not hasattr(pipeline, "predict_proba"):
        raise TypeError(
            "This pipeline has no predict_proba. Calibrate margin-based SVMs "
            "inside the training pipeline before threshold selection."
        )
    values = np.asarray(pipeline.predict_proba(texts))
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    return values


def _score_binary(y_true, pred, metric="f1", beta=1.0):
    """Score one binary label for threshold selection."""
    if metric == "f_beta":
        from sklearn.metrics import fbeta_score
        return float(fbeta_score(y_true, pred, beta=beta, zero_division=0))
    if metric == "recall":
        from sklearn.metrics import recall_score
        return float(recall_score(y_true, pred, zero_division=0))
    # Default: F1, a balanced harmful-content objective.
    return float(f1_score(y_true, pred, zero_division=0))


def choose_label_threshold(y_true, probabilities, lo=0.15, hi=0.85, step=0.01,
                           metric="f1", beta=1.0):
    """Select one threshold for ONE label from development OOF predictions."""
    best_threshold, best_score = 0.50, -1.0
    best_stats = None
    for threshold in np.arange(lo, hi + step / 2, step):
        pred = (probabilities >= threshold).astype(int)
        score = _score_binary(y_true, pred, metric=metric, beta=beta)
        p, r, f1, support = precision_recall_fscore_support(
            y_true, pred, average="binary", zero_division=0)
        candidate = (score, -abs(float(threshold) - 0.50))
        current = (best_score, -abs(best_threshold - 0.50))
        if candidate > current:
            best_threshold = float(threshold)
            best_score = float(score)
            best_stats = {
                "precision": float(p),
                "recall": float(r),
                "f1": float(f1),
                "support_positive": int(np.sum(y_true)),
                "support_negative": int(len(y_true) - np.sum(y_true)),
            }
    return round(best_threshold, 2), best_stats or {
        "precision": 0.0, "recall": 0.0, "f1": 0.0,
        "support_positive": int(np.sum(y_true)),
        "support_negative": int(len(y_true) - np.sum(y_true)),
    }


def choose_thresholds(y_true, probabilities, labels):
    """Select six independent OOF thresholds, one per output label.

    The primary ``abusive`` label uses its own configured metric; all target
    labels use F1.  This avoids allowing abundant target labels to dominate
    the harmful-content decision.
    """
    thresholds = {}
    metrics = {}
    for i, label in enumerate(labels):
        if label == PRIMARY_LABEL:
            metric = PRIMARY_THRESHOLD_METRIC
            beta = PRIMARY_F_BETA if metric == "f_beta" else 1.0
        else:
            metric = "f1"
            beta = 1.0
        th, stats = choose_label_threshold(
            np.asarray(y_true)[:, i], np.asarray(probabilities)[:, i],
            metric=metric, beta=beta)
        thresholds[label] = th
        metrics[label] = {"metric": metric, "threshold": th, **stats}
    return thresholds, metrics


def apply_thresholds(probabilities, thresholds, labels):
    """Convert probabilities to binary predictions column-by-column."""
    p = np.asarray(probabilities)
    return np.column_stack([
        (p[:, i] >= float(thresholds.get(label, 0.5))).astype(int)
        for i, label in enumerate(labels)
    ])


def _save_per_label(model_name, y_true, y_pred, labels, thresholds=None):
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0)
    out = pd.DataFrame({
        "label": labels,
        "threshold": [float((thresholds or {}).get(l, 0.5)) for l in labels],
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
    })
    slug = re.sub(r"[^a-z0-9]+", "_", model_name.lower()).strip("_")
    out.to_csv(os.path.join("results", f"per_label_{slug}.csv"), index=False)


def _save_oof_audit(model_name, y_true, probabilities, labels, thresholds, fold_ids):
    """Persist pooled OOF probabilities so threshold choices are auditable."""
    rows = pd.DataFrame(probabilities, columns=[f"prob_{l}" for l in labels])
    for i, l in enumerate(labels):
        rows[f"true_{l}"] = np.asarray(y_true)[:, i]
        rows[f"pred_{l}"] = (probabilities[:, i] >= thresholds[l]).astype(int)
    rows["cv_fold"] = fold_ids
    rows["row_index_in_development"] = np.arange(len(rows))
    slug = re.sub(r"[^a-z0-9]+", "_", model_name.lower()).strip("_")
    rows.to_csv(os.path.join("results", f"oof_predictions_{slug}.csv"), index=False)


def train_and_save(model_name, pipeline, model_path, sample=None):
    X_dev, X_test, y_dev, y_test, labels = prepare_data(sample=sample)
    os.makedirs("results", exist_ok=True)

    fold_key = stratification_key(y_dev, min_count=5)
    splitter = StratifiedKFold(n_splits=3, shuffle=True,
                              random_state=RANDOM_STATE)
    oof_probabilities = np.zeros((len(X_dev), len(labels)), dtype=float)
    oof_fold_ids = np.zeros(len(X_dev), dtype=int)

    print(f"\nThree-fold out-of-fold threshold selection: {model_name}")
    cv_started = time.time()
    for fold, (train_idx, valid_idx) in enumerate(
            splitter.split(X_dev, fold_key), start=1):
        fold_pipeline = clone(pipeline)
        fold_pipeline.fit(X_dev.iloc[train_idx], y_dev.iloc[train_idx])
        oof_probabilities[valid_idx] = label_probabilities(
            fold_pipeline, list(X_dev.iloc[valid_idx]))
        oof_fold_ids[valid_idx] = fold
        print(f"  completed fold {fold}/3")

    thresholds, threshold_metrics = choose_thresholds(
        y_dev.values, oof_probabilities, labels)
    oof_pred = apply_thresholds(oof_probabilities, thresholds, labels)
    cv_f1_micro = float(f1_score(y_dev.values, oof_pred, average="micro", zero_division=0))
    primary_idx = labels.index(PRIMARY_LABEL)
    primary_oof_f1 = float(f1_score(
        y_dev.values[:, primary_idx], oof_pred[:, primary_idx], zero_division=0))
    primary_oof_p, primary_oof_r, _, _ = precision_recall_fscore_support(
        y_dev.values[:, primary_idx], oof_pred[:, primary_idx],
        average="binary", zero_division=0)
    fold_micro = []
    for fold in range(1, 4):
        idx = np.flatnonzero(oof_fold_ids == fold)
        fold_micro.append(float(f1_score(
            y_dev.iloc[idx].values, oof_pred[idx], average="micro", zero_division=0)))
    cv_time = time.time() - cv_started

    print("Selected OOF thresholds:")
    for label in labels:
        m = threshold_metrics[label]
        print(f"  {label:20s} threshold={thresholds[label]:.2f} "
              f"P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} ({m['metric']})")
    print(f"OOF micro-F1: {cv_f1_micro:.4f}; primary abusive F1={primary_oof_f1:.4f}")

    _save_oof_audit(model_name, y_dev.values, oof_probabilities,
                    labels, thresholds, oof_fold_ids)

    print(f"\nTraining final {model_name} on the complete 80% development set ...")
    started = time.time()
    pipeline.fit(X_dev, y_dev)
    train_time = time.time() - started

    started = time.time()
    test_probabilities = label_probabilities(pipeline, list(X_test))
    y_pred = apply_thresholds(test_probabilities, thresholds, labels)
    predict_time = time.time() - started

    result = evaluate_model(model_name, y_test.values, y_pred, labels,
                            train_time=train_time, predict_time=predict_time,
                            probabilities=test_probabilities, thresholds=thresholds)
    result.update({
        "primary_threshold": thresholds[PRIMARY_LABEL],
        "threshold_selection_metric": PRIMARY_THRESHOLD_METRIC,
        "primary_f_beta": PRIMARY_F_BETA if PRIMARY_THRESHOLD_METRIC == "f_beta" else "",
        "thresholds": "; ".join(f"{l}={thresholds[l]:.2f}" for l in labels),
        "cv_f1_micro_mean": round(float(np.mean(fold_micro)), 4),
        "cv_f1_micro_std": round(float(np.std(fold_micro, ddof=1)), 4),
        "oof_f1_micro": round(cv_f1_micro, 4),
        "oof_primary_precision": round(float(primary_oof_p), 4),
        "oof_primary_recall": round(float(primary_oof_r), 4),
        "oof_primary_f1": round(primary_oof_f1, 4),
        "cv_selection_time_sec": round(cv_time, 3),
    })
    save_result(result)
    _save_per_label(model_name, y_test.values, y_pred, labels, thresholds)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    bundle = {
        "pipeline": pipeline,
        "labels": labels,
        "thresholds": thresholds,
        # Keep scalar key for backwards compatibility, but it is now only the
        # primary harmful threshold, not a global multilabel threshold.
        "threshold": thresholds[PRIMARY_LABEL],
        "primary_threshold": thresholds[PRIMARY_LABEL],
        "primary_threshold_metric": PRIMARY_THRESHOLD_METRIC,
        "primary_threshold_beta": PRIMARY_F_BETA,
        "threshold_metrics_oof": threshold_metrics,
        "threshold_source": "5-fold pooled out-of-fold predictions on the 80% development set; final 20% test set untouched",
        "cv_fold_f1_micro": fold_micro,
        "oof_micro_f1": cv_f1_micro,
        "oof_predictions_file": os.path.join(
            "results", f"oof_predictions_{re.sub(r'[^a-z0-9]+', '_', model_name.lower()).strip('_')}.csv"),
    }
    joblib.dump(bundle, model_path, compress=3)
    print(f"Saved trained model -> {model_path}")
    return pipeline, thresholds


def retune_saved_bundle(model_name, model_path, sample=None):
    """Retune per-label thresholds for an existing trained pipeline.

    This is useful when a previous final pipeline was already trained with the
    same feature/model configuration. It creates fresh OOF predictions on the
    80% development split, selects six thresholds, evaluates the existing
    pipeline once on the untouched final test split, and rewrites the bundle
    with the audited threshold metadata. No test labels are used for tuning.
    """
    X_dev, X_test, y_dev, y_test, labels = prepare_data(sample=sample)
    os.makedirs("results", exist_ok=True)
    bundle = joblib.load(model_path)
    pipeline = bundle["pipeline"]

    fold_key = stratification_key(y_dev, min_count=5)
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    oof_probabilities = np.zeros((len(X_dev), len(labels)), dtype=float)
    oof_fold_ids = np.zeros(len(X_dev), dtype=int)
    started = time.time()
    for fold, (train_idx, valid_idx) in enumerate(splitter.split(X_dev, fold_key), start=1):
        fold_pipeline = clone(pipeline)
        fold_pipeline.fit(X_dev.iloc[train_idx], y_dev.iloc[train_idx])
        oof_probabilities[valid_idx] = label_probabilities(
            fold_pipeline, list(X_dev.iloc[valid_idx]))
        oof_fold_ids[valid_idx] = fold
        print(f"  completed fold {fold}/3")

    thresholds, threshold_metrics = choose_thresholds(
        y_dev.values, oof_probabilities, labels)
    oof_pred = apply_thresholds(oof_probabilities, thresholds, labels)
    oof_micro = float(f1_score(y_dev.values, oof_pred, average="micro", zero_division=0))
    primary_idx = labels.index(PRIMARY_LABEL)
    primary_oof = _score_binary(y_dev.values[:, primary_idx],
                                oof_pred[:, primary_idx], metric=PRIMARY_THRESHOLD_METRIC,
                                beta=PRIMARY_F_BETA)
    _save_oof_audit(model_name, y_dev.values, oof_probabilities, labels, thresholds, oof_fold_ids)

    test_probabilities = label_probabilities(pipeline, list(X_test))
    y_pred = apply_thresholds(test_probabilities, thresholds, labels)
    result = evaluate_model(model_name, y_test.values, y_pred, labels,
                            probabilities=test_probabilities, thresholds=thresholds)
    result.update({
        "primary_threshold": thresholds[PRIMARY_LABEL],
        "threshold_selection_metric": PRIMARY_THRESHOLD_METRIC,
        "primary_f_beta": PRIMARY_F_BETA if PRIMARY_THRESHOLD_METRIC == "f_beta" else "",
        "thresholds": "; ".join(f"{l}={thresholds[l]:.2f}" for l in labels),
        "oof_f1_micro": round(oof_micro, 4),
        "oof_primary_selection_score": round(float(primary_oof), 4),
        "cv_selection_time_sec": round(time.time() - started, 3),
    })
    save_result(result)
    _save_per_label(model_name, y_test.values, y_pred, labels, thresholds)

    bundle.update({
        "labels": labels,
        "thresholds": thresholds,
        "threshold": thresholds[PRIMARY_LABEL],
        "primary_threshold": thresholds[PRIMARY_LABEL],
        "primary_threshold_metric": PRIMARY_THRESHOLD_METRIC,
        "primary_threshold_beta": PRIMARY_F_BETA,
        "threshold_metrics_oof": threshold_metrics,
        "threshold_source": "3-fold pooled out-of-fold predictions on the 80% development set; final 20% test set untouched",
        "oof_micro_f1": oof_micro,
        "oof_predictions_file": os.path.join("results", f"oof_predictions_{re.sub(r'[^a-z0-9]+', '_', model_name.lower()).strip('_')}.csv"),
    })
    joblib.dump(bundle, model_path, compress=3)
    return bundle, result

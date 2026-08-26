"""Leakage-safe shared training utilities.

The project uses:
1. An untouched 20% final test set.
2. Development-only model selection by five-fold cross-validation.
3. Five-fold out-of-fold (OOF) probabilities on the full development set.
4. One independently selected threshold per label from OOF predictions.
5. A primary ``abusive`` threshold used exclusively for the final YES/NO verdict.
6. A final fit on all 80% development data, followed by one final test evaluation.

No model configuration or threshold is selected from the final test set.
"""
import os
import re
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    f1_score, precision_score, recall_score, confusion_matrix,
    precision_recall_fscore_support
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import FeatureUnion

from .common import prepare_data, stratification_key, RANDOM_STATE
from .evaluate import evaluate_model, save_result


LABEL_NAMES = ["abusive", "Race", "Religion", "Gender",
               "Sexual_Orientation", "Miscellaneous"]


def build_word_char_features(word_max_features=30000, char_max_features=10000,
                             word_ngram_range=(1, 3)):
    """TF-IDF with word unigrams/bigrams/trigrams plus character n-grams."""
    # Word trigrams preserve phrases such as "go back to" and "do not belong".
    # Character 3-5 grams complement them by retaining partial patterns under
    # punctuation insertion and spelling obfuscation.
    return FeatureUnion([
        ("word", TfidfVectorizer(
            max_features=word_max_features,
            ngram_range=word_ngram_range,
            min_df=2,
            sublinear_tf=True)),
        ("char", TfidfVectorizer(
            max_features=char_max_features,
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=2,
            sublinear_tf=True)),
    ])


def label_probabilities(pipeline, texts):
    """Return genuine fitted probabilities; never fabricate them from margins."""
    if not hasattr(pipeline, "predict_proba"):
        raise TypeError(
            "This pipeline has no predict_proba. Calibrate margin-based models "
            "inside the training pipeline before using probability thresholds."
        )
    values = np.asarray(pipeline.predict_proba(texts))
    if values.ndim == 3:
        values = values[:, :, 1]
    return values.reshape(len(texts), -1)


def _primary_stats(y_true, pred):
    y_true = np.asarray(y_true).astype(int)
    pred = np.asarray(pred).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    precision = precision_score(y_true, pred, zero_division=0)
    recall = recall_score(y_true, pred, zero_division=0)
    f1 = f1_score(y_true, pred, zero_division=0)
    fpr = fp / max(fp + tn, 1)
    fnr = fn / max(fn + tp, 1)
    return {
        "precision": float(precision), "recall": float(recall),
        "f1": float(f1), "fpr": float(fpr), "fnr": float(fnr),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)
    }


def choose_primary_threshold(y_true, probabilities, lo=0.15, hi=0.85, step=0.01):
    """Choose the abusive threshold using ONLY the primary OOF predictions.

    The objective is harmful-content F1. When several thresholds are effectively
    tied, the least aggressive one (lowest benign false-positive rate, then
    highest precision) is preferred. This prevents a tiny F1 fluctuation from
    selecting an unnecessarily permissive threshold.
    """
    y = np.asarray(y_true).astype(int)
    p = np.asarray(probabilities, dtype=float)
    candidates = []
    for threshold in np.arange(lo, hi + step / 2, step):
        stats = _primary_stats(y, (p >= threshold).astype(int))
        candidates.append((float(threshold), stats))
    # Harmful-content F1 is the primary objective, but a moderation system
    # should not choose an extremely aggressive point simply because recall is
    # high. Prefer F1-optimal candidates that also reach a minimum precision.
    # If the precision floor is not achievable, fall back to the unconstrained
    # F1 optimum and report the resulting precision/recall/FPR.
    precision_floor = 0.75
    eligible = [x for x in candidates if x[1]["precision"] >= precision_floor]
    pool = eligible if eligible else candidates
    best_f1 = max(x[1]["f1"] for x in pool)
    best = [x for x in pool if abs(x[1]["f1"] - best_f1) <= 1e-12]
    threshold, stats = sorted(
        best, key=lambda x: (-x[1]["precision"], x[1]["fpr"], x[0])
    )[0]
    return round(threshold, 2), stats


def choose_label_threshold(y_true, probabilities, lo=0.15, hi=0.85, step=0.01):
    """Choose one threshold for one label using that label's OOF F1 only."""
    best_threshold, best_f1 = 0.50, -1.0
    for threshold in np.arange(lo, hi + step / 2, step):
        score = f1_score(y_true, probabilities >= threshold, zero_division=0)
        if score > best_f1 + 1e-12:
            best_threshold, best_f1 = float(threshold), float(score)
    return round(best_threshold, 2), float(best_f1)


def choose_thresholds(y_true, probabilities, labels):
    """Return a six-label threshold dictionary and OOF evidence."""
    thresholds = {}
    evidence = []
    for i, label in enumerate(labels):
        if label == "abusive":
            th, stats = choose_primary_threshold(
                y_true[:, i], probabilities[:, i])
            thresholds[label] = th
            evidence.append({
                "label": label, "threshold": th,
                "objective": "harmful-content F1 with minimum precision floor 0.75",
                "f1": stats["f1"], "precision": stats["precision"],
                "recall": stats["recall"], "false_positive_rate": stats["fpr"],
                "false_negative_rate": stats["fnr"],
                "tn": stats["tn"], "fp": stats["fp"],
                "fn": stats["fn"], "tp": stats["tp"],
            })
        else:
            th, f1 = choose_label_threshold(y_true[:, i], probabilities[:, i])
            thresholds[label] = th
            pred = (probabilities[:, i] >= th).astype(int)
            stats = _primary_stats(y_true[:, i], pred)
            evidence.append({
                "label": label, "threshold": th,
                "objective": "label-specific F1",
                "f1": f1, "precision": stats["precision"],
                "recall": stats["recall"],
                "false_positive_rate": stats["fpr"],
                "false_negative_rate": stats["fnr"],
                "tn": stats["tn"], "fp": stats["fp"],
                "fn": stats["fn"], "tp": stats["tp"],
            })
    return thresholds, pd.DataFrame(evidence)


def _save_oof_evidence(model_name, y_true, probabilities, labels, thresholds):
    slug = re.sub(r"[^a-z0-9]+", "_", model_name.lower()).strip("_")
    out = pd.DataFrame({"row_index": np.arange(len(y_true))})
    for i, label in enumerate(labels):
        out[f"true_{label}"] = y_true[:, i].astype(int)
        out[f"prob_{label}"] = probabilities[:, i]
        out[f"pred_{label}"] = (
            probabilities[:, i] >= thresholds[label]).astype(int)
    out.to_csv(os.path.join("results", f"oof_{slug}.csv"), index=False)
    return os.path.join("results", f"oof_{slug}.csv")


def _save_per_label(model_name, y_true, y_pred, labels, thresholds):
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0)
    rows = []
    for i, label in enumerate(labels):
        stats = _primary_stats(y_true[:, i], y_pred[:, i])
        rows.append({
            "label": label,
            "threshold": thresholds[label],
            "precision": precision[i],
            "recall": recall[i],
            "f1": f1[i],
            "support": support[i],
            "false_positive_rate": stats["fpr"],
            "false_negative_rate": stats["fnr"],
            "tn": stats["tn"], "fp": stats["fp"],
            "fn": stats["fn"], "tp": stats["tp"],
        })
    slug = re.sub(r"[^a-z0-9]+", "_", model_name.lower()).strip("_")
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join("results", f"per_label_{slug}.csv"), index=False)


def _primary_cv_score(y_true, probabilities):
    """Development-only model-selection score.

    Primary F1 is the main objective. FPR is reported/used as a small
    tie-break so a configuration that flags almost every benign comment is
    not preferred when F1 is essentially tied.
    """
    stats = _primary_stats(y_true, (probabilities >= 0.50).astype(int))
    return stats["f1"], stats["fpr"]


def select_model_configuration(model_name, pipeline, X_dev, y_dev, candidates):
    """Select model hyperparameters using only development-side CV.

    ``candidates`` is a list of fully constructed pipelines. The final test
    set is never visible here. Five-fold cross-validation is used consistently
    for development-side model configuration selection.
    """
    # The complete 80% development partition is used for model-selection CV.
    # y_sel is explicitly aligned with X_sel so fold indices can never be
    # accidentally applied to a different label table.
    X_sel = X_dev.reset_index(drop=True)
    y_sel = y_dev.reset_index(drop=True)
    key = stratification_key(y_sel, min_count=5)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    best_pipeline = None
    best_key = (-1.0, -1.0, float("inf"))

    print(f"\nDevelopment-only model configuration search: {model_name}")
    for candidate_name, candidate in candidates:
        fold_scores = []
        fold_fprs = []
        for train_idx, valid_idx in splitter.split(X_sel, key):
            fitted = clone(candidate)
            fitted.fit(X_sel.iloc[train_idx], y_sel.iloc[train_idx])
            prob = label_probabilities(
                fitted, list(X_sel.iloc[valid_idx]))[:, 0]
            stats = _primary_stats(
                y_sel.iloc[valid_idx].values[:, 0],
                (prob >= 0.50).astype(int))
            fold_scores.append(stats["f1"])
            fold_fprs.append(stats["fpr"])
        mean_f1 = float(np.mean(fold_scores))
        mean_fpr = float(np.mean(fold_fprs))
        # Model configuration is selected with a primary harmful-content
        # utility that rewards F1 while explicitly penalising benign false
        # positives. This prevents a configuration that catches almost
        # everything by flagging nearly every benign comment from winning.
        utility = mean_f1 - 0.50 * mean_fpr
        rows.append({
            "configuration": candidate_name,
            "primary_f1_cv_mean": mean_f1,
            "primary_f1_cv_std": float(np.std(fold_scores, ddof=1)),
            "benign_false_positive_rate_cv_mean": mean_fpr,
            "selection_utility_f1_minus_0.5_fpr": utility,
        })
        key_value = (utility, mean_f1, -mean_fpr)
        current_key = (best_key[0], best_key[1], -best_key[2])
        if key_value > current_key:
            best_key = (utility, mean_f1, mean_fpr)
            best_pipeline = candidate
        print(f"  {candidate_name}: F1={mean_f1:.4f}, FPR={mean_fpr:.4f}")

    os.makedirs("results", exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "_", model_name.lower()).strip("_")
    pd.DataFrame(rows).to_csv(
        os.path.join("results", f"model_selection_{slug}.csv"), index=False)
    return clone(best_pipeline), pd.DataFrame(rows)


def train_and_save(model_name, pipeline, model_path, sample=None,
                   candidates=None):
    X_dev, X_test, y_dev, y_test, labels = prepare_data(sample=sample)
    os.makedirs("results", exist_ok=True)

    selection_rows = None
    if candidates:
        pipeline, selection_rows = select_model_configuration(
            model_name, pipeline, X_dev, y_dev, candidates)

    # Threshold selection uses the COMPLETE 80% development set. Every row
    # receives a prediction from a fold that did not train on that row.
    X_oof = X_dev.reset_index(drop=True)
    y_oof = y_dev.reset_index(drop=True)
    fold_key = stratification_key(y_oof, min_count=5)
    splitter = StratifiedKFold(n_splits=5, shuffle=True,
                               random_state=RANDOM_STATE)
    oof_probabilities = np.zeros((len(X_oof), len(labels)), dtype=float)
    fold_f1 = []
    cv_started = time.time()

    print(f"\nFive-fold OOF threshold selection on the full 80% development set: {model_name}")
    for fold, (train_idx, valid_idx) in enumerate(
            splitter.split(X_oof, fold_key), start=1):
        fold_pipeline = clone(pipeline)
        fold_pipeline.fit(X_oof.iloc[train_idx], y_oof.iloc[train_idx])
        p = label_probabilities(
            fold_pipeline, list(X_oof.iloc[valid_idx]))
        oof_probabilities[valid_idx] = p
        fold_pred = np.column_stack([
            p[:, i] >= 0.50
            for i, label in enumerate(labels)
        ]).astype(int)
        fold_f1.append(float(f1_score(
            y_oof.iloc[valid_idx].values, fold_pred, average="micro", zero_division=0)))
        print(f"  completed fold {fold}/5")

    thresholds, threshold_evidence = choose_thresholds(
        y_oof.values, oof_probabilities, labels)
    threshold_evidence.to_csv(
        os.path.join("results", f"threshold_evidence_{re.sub(r'[^a-z0-9]+','_',model_name.lower()).strip('_')}.csv"),
        index=False)
    oof_path = _save_oof_evidence(
        model_name, y_oof.values, oof_probabilities, labels, thresholds)

    primary_oof = threshold_evidence.loc[
        threshold_evidence["label"] == "abusive"].iloc[0]
    fold_time = time.time() - cv_started
    print(f"Selected thresholds: {thresholds}")
    print(
        f"OOF abusive F1={primary_oof['f1']:.4f}, "
        f"precision={primary_oof['precision']:.4f}, "
        f"recall={primary_oof['recall']:.4f}, "
        f"FPR={primary_oof['false_positive_rate']:.4f}"
    )

    print(f"\nTraining final {model_name} on complete 80% development data ...")
    started = time.time()
    pipeline.fit(X_dev, y_dev)
    train_time = time.time() - started

    started = time.time()
    test_probabilities = label_probabilities(pipeline, list(X_test))
    y_pred = np.column_stack([
        test_probabilities[:, i] >= thresholds[label]
        for i, label in enumerate(labels)
    ]).astype(int)
    predict_time = time.time() - started

    result = evaluate_model(
        model_name, y_test.values, y_pred, labels,
        train_time=train_time, predict_time=predict_time,
        primary_probabilities=test_probabilities[:, 0],
        primary_threshold=thresholds["abusive"]
    )
    result.update({
        "threshold_abusive": thresholds["abusive"],
        "oof_primary_f1": round(float(primary_oof["f1"]), 4),
        "oof_primary_precision": round(float(primary_oof["precision"]), 4),
        "oof_primary_recall": round(float(primary_oof["recall"]), 4),
        "oof_primary_fpr": round(float(primary_oof["false_positive_rate"]), 4),
        "oof_selection_time_sec": round(fold_time, 3),
        "cv_fold_f1_micro_mean": round(float(np.mean(fold_f1)), 4),
        "cv_fold_f1_micro_std": round(float(np.std(fold_f1, ddof=1)), 4),
    })
    save_result(result)
    _save_per_label(model_name, y_test.values, y_pred, labels, thresholds)

    bundle = {
        "pipeline": pipeline,
        "labels": labels,
        "thresholds": thresholds,
        "threshold": thresholds["abusive"],  # compatibility with older code
        "primary_label": "abusive",
        "threshold_source": (
            "Five-fold OOF predictions on the full 80% development set; "
            "abusive selected by primary harmful-content F1"
        ),
        "threshold_objective": {
            "abusive": "harmful-content F1 with benign-FPR/precision tie-break",
            "targets": "independent label-specific F1",
        },
        "oof_evidence_file": oof_path,
        "cv_fold_f1_micro": fold_f1,
        "model_selection": (
            selection_rows.to_dict("records")
            if selection_rows is not None else []
        ),
    }
    joblib.dump(bundle, model_path, compress=3)
    print(f"Saved trained model -> {model_path}")
    return pipeline

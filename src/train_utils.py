"""Shared training/evaluation utilities using leakage-safe 80/20 + CV."""
import os
import time
import joblib
import numpy as np
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from .common import prepare_data, RANDOM_STATE
from .evaluate import evaluate_model, save_result

CV_SPLITS = 2


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


def _scores(pipeline, texts):
    texts = list(texts)
    try:
        return np.asarray(pipeline.predict_proba(texts))
    except (AttributeError, RuntimeError):
        d = np.asarray(pipeline.decision_function(texts))
        if d.ndim == 1:
            d = d.reshape(-1, 1)
        return 1.0 / (1.0 + np.exp(-d))


def select_threshold_cv(pipeline, X_train, y_train, n_splits=CV_SPLITS,
                        step=0.02, lo=0.20, hi=0.86):
    """Select a single model-specific threshold using out-of-fold training scores.

    The full 20% final test set is never passed to this function.
    Each fold clones and fits the entire text pipeline on that fold's training data,
    so TF-IDF vocabulary/statistics are also learned fold-by-fold.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    oof_scores = np.zeros((len(X_train), y_train.shape[1]), dtype=float)
    seen = np.zeros(len(X_train), dtype=bool)

    X_arr = np.asarray(X_train)
    y_arr = y_train.values
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_arr), start=1):
        fold_model = clone(pipeline)
        fold_model.fit(X_arr[tr_idx], y_arr[tr_idx])
        oof_scores[va_idx] = _scores(fold_model, X_arr[va_idx])
        seen[va_idx] = True
        print(f"CV fold {fold}/{n_splits} complete")

    if not seen.all():
        raise RuntimeError("Cross-validation did not generate out-of-fold predictions for every training example.")

    best_th, best_f1 = 0.50, -1.0
    for th in np.arange(lo, hi + 1e-9, step):
        pred = (oof_scores >= th).astype(int)
        f1 = f1_score(y_arr, pred, average="micro", zero_division=0)
        if f1 > best_f1:
            best_th, best_f1 = float(th), float(f1)
    return round(best_th, 2), float(best_f1)


def train_and_save(model_name, pipeline, model_path, sample=None):
    X_train, X_test, y_train, y_test, labels = prepare_data(sample=sample)

    print(f"\nSelecting {model_name} threshold using {CV_SPLITS}-fold cross-validation on the 80% training data ...")
    cv_t0 = time.time()
    threshold, cv_f1 = select_threshold_cv(pipeline, X_train, y_train)
    cv_time = time.time() - cv_t0
    print(f"CV-selected threshold: {threshold:.2f} (OOF micro-F1={cv_f1:.4f}, CV time={cv_time:.2f}s)")

    print(f"\nFitting final {model_name} on the full 80% training data ...")
    t0 = time.time()
    pipeline.fit(X_train, y_train)
    train_time = time.time() - t0

    t0 = time.time()
    test_scores = _scores(pipeline, X_test)
    y_pred = (test_scores >= threshold).astype(int)
    predict_time = time.time() - t0

    result = evaluate_model(model_name, y_test.values, y_pred, labels,
                            train_time=train_time, predict_time=predict_time,
                            threshold=threshold, validation_f1=cv_f1)
    result["cv_splits"] = CV_SPLITS
    save_result(result)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump({"pipeline": pipeline, "labels": labels,
                 "threshold": threshold,
                 "cv_micro_f1": cv_f1,
                 "cv_splits": CV_SPLITS}, model_path, compress=3)
    print(f"Saved trained model -> {model_path}")
    return pipeline

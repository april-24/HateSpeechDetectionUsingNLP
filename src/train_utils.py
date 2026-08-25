"""
train_utils.py
--------------
Shared routine used by all three model scripts.

The routine enforces a leakage-safe 60/20/20 train/validation/test protocol:
the pipeline is fitted only on training data; the validation set is used to
select the model-specific decision threshold; the final test set is evaluated
once using that fixed threshold.
"""
import os
import time
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.metrics import f1_score
from .common import prepare_data
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

def _scores(pipeline, texts):
    try:
        return np.asarray(pipeline.predict_proba(texts))
    except (AttributeError, RuntimeError):
        d = np.asarray(pipeline.decision_function(texts))
        if d.ndim == 1:
            d = d.reshape(1, -1)
        return 1.0 / (1.0 + np.exp(-d))

def select_threshold(pipeline, X_val, y_val, step=0.02, lo=0.20, hi=0.86):
    """Select threshold using validation data only."""
    P = _scores(pipeline, list(X_val))
    best_th, best_f1 = 0.50, -1.0
    for th in np.arange(lo, hi + 1e-9, step):
        pred = (P >= th).astype(int)
        f1 = f1_score(y_val.values, pred, average="micro", zero_division=0)
        if f1 > best_f1:
            best_th, best_f1 = float(th), float(f1)
    return round(best_th, 2), float(best_f1)

def train_and_save(model_name, pipeline, model_path, sample=None):
    X_train, X_val, X_test, y_train, y_val, y_test, labels = prepare_data(sample=sample)

    print(f"\nTraining {model_name} ...")
    t0 = time.time()
    pipeline.fit(X_train, y_train)
    train_time = time.time() - t0

    threshold, val_f1 = select_threshold(pipeline, X_val, y_val)
    print(f"Validation-selected threshold: {threshold:.2f} (validation micro-F1={val_f1:.4f})")

    t0 = time.time()
    test_scores = _scores(pipeline, list(X_test))
    y_pred = (test_scores >= threshold).astype(int)
    predict_time = time.time() - t0

    result = evaluate_model(model_name, y_test.values, y_pred, labels,
                            train_time=train_time, predict_time=predict_time,
                            threshold=threshold, validation_f1=val_f1)
    save_result(result)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump({"pipeline": pipeline, "labels": labels,
                 "threshold": threshold,
                 "validation_micro_f1": val_f1}, model_path, compress=3)
    print(f"Saved trained model -> {model_path}")
    return pipeline


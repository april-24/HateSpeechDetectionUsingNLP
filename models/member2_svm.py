"""Member 2: calibrated Linear SVM + TF-IDF.

LinearSVC itself does not output probabilities. CalibratedClassifierCV performs
formal sigmoid calibration inside the training data, so probability thresholds
can be selected from OOF predictions without treating raw margins as
probabilities.
"""
import os
import sys
import argparse
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.train_utils import train_and_save, build_word_char_features
from src.config import RESULTS_DIR

MODEL_NAME = "Linear SVM"
MODEL_PATH = str(RESULTS_DIR / "model_svm.joblib")


def build_pipeline(C=1.0, class_weight="balanced"):
    return Pipeline([
        ("features", build_word_char_features(
            word_max_features=5000,
            char_max_features=500,
            word_ngram_range=(1, 3))),
        ("clf", OneVsRestClassifier(
            CalibratedClassifierCV(
                LinearSVC(C=C, class_weight=class_weight, max_iter=5000),
                method="sigmoid", cv=3
            )
        ))
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=None)
    args = ap.parse_args()
    candidates = [
        ("C=0.5, class_weight=balanced", build_pipeline(0.5, "balanced")),
        ("C=1.0, class_weight=balanced", build_pipeline(1.0, "balanced")),
        ("C=2.0, class_weight=balanced", build_pipeline(2.0, "balanced")),
        ("C=1.0, class_weight=None", build_pipeline(1.0, None)),
    ]
    train_and_save(
        MODEL_NAME, candidates[1][1], MODEL_PATH,
        sample=args.sample, candidates=candidates
    )


if __name__ == "__main__":
    main()

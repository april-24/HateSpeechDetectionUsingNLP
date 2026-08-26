"""
member2_svm.py   (MEMBER 2's solution)
======================================
Method : Linear Support Vector Machine (LinearSVC) + TF-IDF features (word + character n-grams)
Wrapper: OneVsRestClassifier (one SVM per label -> multi-label)

An SVM finds the separating hyperplane with the widest margin between classes.
On sparse high-dimensional TF-IDF text this linear SVM is a strong, fast baseline.
The LinearSVC is wrapped in cross-validated sigmoid calibration so that the
application can display predicted probabilities without pretending that a
simple logistic transform of the raw margin is calibrated.

Run from the project root:
    python -m models.member2_svm
    python -m models.member2_svm --sample 20000
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

MODEL_NAME = "Linear SVM"
MODEL_PATH = "results/model_svm.joblib"


def build_pipeline():
    return Pipeline([
        ("features", build_word_char_features(word_max_features=30000,
                                              char_max_features=10000)),
        ("clf", OneVsRestClassifier(CalibratedClassifierCV(
            LinearSVC(C=1.0, class_weight="balanced"), method="sigmoid", cv=3))),
    ])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=None)
    args = ap.parse_args()
    train_and_save(MODEL_NAME, build_pipeline(), MODEL_PATH, sample=args.sample)

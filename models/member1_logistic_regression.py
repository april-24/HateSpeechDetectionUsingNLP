"""
member1_logistic_regression.py   (MEMBER 1's solution)
======================================================
Method : Logistic Regression + TF-IDF features (word + character n-grams)
Wrapper: OneVsRestClassifier (one LR per label -> multi-label)

Logistic Regression models the probability of each label with a linear decision
boundary over the TF-IDF features. It is fast, probability estimates, and its learned
weights are easy to interpret (used later to highlight influential words).

Run from the project root:
    python -m models.member1_logistic_regression
    python -m models.member1_logistic_regression --sample 20000
"""

import os
import sys
import argparse
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.train_utils import train_and_save, build_word_char_features

MODEL_NAME = "Logistic Regression"
MODEL_PATH = "results/model_lr.joblib"


def build_pipeline():
    return Pipeline([
        ("features", build_word_char_features(word_max_features=30000,
                                              char_max_features=10000)),
        ("clf", OneVsRestClassifier(
            LogisticRegression(max_iter=1000, C=3.0, class_weight="balanced"))),
    ])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=None)
    args = ap.parse_args()
    train_and_save(MODEL_NAME, build_pipeline(), MODEL_PATH, sample=args.sample)

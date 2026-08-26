"""Member 1: Logistic Regression + TF-IDF (word 1-3 grams + character 3-5 grams)."""
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


def build_pipeline(C=1.0, class_weight="balanced"):
    return Pipeline([
        ("features", build_word_char_features(
            word_max_features=8000,
            char_max_features=1000,
            word_ngram_range=(1, 3))),
        ("clf", OneVsRestClassifier(
            LogisticRegression(
                max_iter=1200, C=C, class_weight=class_weight,
                solver="saga", n_jobs=-1
            )))
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=None)
    args = ap.parse_args()
    candidates = [
        ("C=0.5, class_weight=balanced", build_pipeline(0.5, "balanced")),
        ("C=1.0, class_weight=balanced", build_pipeline(1.0, "balanced")),
        ("C=2.0, class_weight=balanced", build_pipeline(2.0, "balanced")),
    ]
    train_and_save(
        MODEL_NAME, candidates[1][1], MODEL_PATH,
        sample=args.sample, candidates=candidates
    )


if __name__ == "__main__":
    main()

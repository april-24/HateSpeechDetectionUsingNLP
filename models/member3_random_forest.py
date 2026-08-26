"""Member 3: Random Forest + TF-IDF word 1-3 grams.

Development-only CV evaluates tree count, depth, leaf size and class weighting.
"""
import os
import sys
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.train_utils import train_and_save
from src.config import RESULTS_DIR

MODEL_NAME = "Random Forest"
MODEL_PATH = str(RESULTS_DIR / "model_rf.joblib")


def build_pipeline(n_estimators=120, max_depth=35, min_samples_leaf=2,
                   class_weight="balanced"):
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=5000, ngram_range=(1, 3),
            min_df=2, sublinear_tf=True)),
        ("clf", OneVsRestClassifier(
            RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                class_weight=class_weight,
                n_jobs=-1,
                random_state=42
            )
        ))
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=None)
    args = ap.parse_args()
    candidates = [
        ("trees=60, depth=20, leaf=1, balanced",
         build_pipeline(60, 20, 1, "balanced")),
        ("trees=90, depth=35, leaf=2, balanced",
         build_pipeline(90, 35, 2, "balanced")),
        ("trees=120, depth=45, leaf=3, none",
         build_pipeline(120, 45, 3, None)),
    ]
    train_and_save(
        MODEL_NAME, candidates[1][1], MODEL_PATH,
        sample=args.sample, candidates=candidates
    )


if __name__ == "__main__":
    main()

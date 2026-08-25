"""
member3_random_forest.py   (MEMBER 3's solution)
================================================
Method : Random Forest + TF-IDF features (word unigrams)
Wrapper: OneVsRestClassifier (one forest per label -> multi-label)

A Random Forest is an ensemble of decision trees; each tree votes and the forest
averages them, which reduces overfitting. Unlike the linear models it can capture
non-linear combinations of words. It provides predict_proba (used for confidence)
and per-feature importances (used to highlight influential words).

REVERTED to this smaller, word-unigram-only configuration after testing showed
adding word bigrams + character n-grams (~33MB model) did NOT fix Random
Forest's core weakness - it still under-detects even obvious cases like slurs
and profanity, because the issue is structural (each split only sees a random
subset of features, and short comments often don't reach the words that
matter across many trees) rather than a feature-richness problem. This smaller
configuration gives essentially the same real-world detection behaviour at
~6-7 MB instead of ~33 MB and trains in seconds instead of ~90 seconds - so
there was no reason to keep paying the larger cost.

A KNOWN, DOCUMENTED LIMITATION (worth citing in your report): Random Forest's
predict_proba on sparse, high-dimensional TF-IDF text tends to be
systematically more conservative (lower) than a linear or probabilistic
model's, even when directionally correct. We tested several fixes - richer
features, probability calibration (CalibratedClassifierCV), higher per-split
feature sampling, and TruncatedSVD dimensionality reduction - none solved it
without a worse trade-off elsewhere (bigger files, much slower training, or
hurting minority-category recall). This is genuinely useful content for a
"strengths & weaknesses" discussion in your documentation. See also
models/naive_bayes_extra.py for a probabilistic model that does not have
this specific weakness.

Run from the project root:
    python -m models.member3_random_forest
    python -m models.member3_random_forest --sample 20000
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

MODEL_NAME = "Random Forest"
MODEL_PATH = "results/model_rf.joblib"


def build_pipeline():
    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=8000, ngram_range=(1, 1),
                                  min_df=3, sublinear_tf=True)),
        ("clf", OneVsRestClassifier(RandomForestClassifier(
            n_estimators=60, max_depth=45, min_samples_leaf=3,
            n_jobs=-1, class_weight="balanced", random_state=42))),
    ])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=None)
    args = ap.parse_args()
    train_and_save(MODEL_NAME, build_pipeline(), MODEL_PATH, sample=args.sample)

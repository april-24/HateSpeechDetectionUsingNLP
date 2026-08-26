"""
compare_models.py
-----------------
Run this AFTER all three members have trained their models. It reads the shared
results/model_scores.csv and produces the comparison table + bar chart used in the
'Results & Discussion' section of the documentation.

    python compare_models.py
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCORES = os.path.join("results", "model_scores.csv")


def main():
    if not os.path.exists(SCORES):
        print("No scores found. Train the models first, e.g.:")
        print("  python -m models.member1_logistic_regression")
        print("  python -m models.member2_svm")
        print("  python -m models.member3_random_forest")
        return

    df = pd.read_csv(SCORES)
    # Order columns nicely
    cols = ["model", "accuracy", "subset_accuracy", "hamming_loss",
            "f1_micro", "f1_macro", "f1_weighted",
            "precision_micro", "recall_micro",
            "train_time_sec", "predict_time_sec"]
    df = df[[c for c in cols if c in df.columns]]

    print("\n================  MODEL COMPARISON  ================\n")
    print(df.to_string(index=False))

    df.to_csv("results/comparison_table.csv", index=False)

    # Bar chart comparing the headline F1 scores
    plt.figure(figsize=(9, 5))
    x = range(len(df))
    width = 0.35
    plt.bar([i - width/2 for i in x], df["f1_micro"], width, label="F1 (micro)",
            color="#c0392b")
    plt.bar([i + width/2 for i in x], df["f1_macro"], width, label="F1 (macro)",
            color="#2980b9")
    plt.xticks(list(x), df["model"], rotation=15, ha="right")
    plt.ylabel("F1 score")
    plt.title("Hate and offensive content detection: model comparison")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/comparison_f1.png", dpi=120)
    plt.close()

    best = df.loc[df["f1_micro"].idxmax(), "model"]
    print(f"\nBest model by micro-F1: {best}")
    print("Saved results/comparison_table.csv and results/comparison_f1.png")


if __name__ == "__main__":
    main()

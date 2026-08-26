"""Build the final comparison table and primary/multilabel F1 figure."""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCORES = os.path.join("results", "model_scores.csv")


def main():
    if not os.path.exists(SCORES):
        raise FileNotFoundError("results/model_scores.csv not found. Finalise all models first.")
    df = pd.read_csv(SCORES)
    preferred = [
        "model", "primary_threshold", "primary_accuracy", "primary_precision",
        "primary_recall", "primary_f1", "primary_false_positive_rate",
        "primary_false_negative_rate", "accuracy", "subset_accuracy",
        "precision_micro", "recall_micro", "f1_micro", "precision_macro",
        "recall_macro", "f1_macro", "train_time_sec", "predict_time_sec"
    ]
    df = df[[c for c in preferred if c in df.columns]]
    df.to_csv("results/comparison_table.csv", index=False)

    x = list(range(len(df)))
    width = 0.24
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar([i - width for i in x], df["primary_f1"], width, label="Primary harmful F1")
    ax.bar(x, df["f1_micro"], width, label="Multilabel micro-F1")
    ax.bar([i + width for i in x], df["f1_macro"], width, label="Multilabel macro-F1")
    ax.set_xticks(x)
    ax.set_xticklabels(df["model"], rotation=15, ha="right")
    ax.set_ylabel("F1 score")
    ax.set_title("Hate and offensive content detection: model comparison")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig("results/comparison_f1.png", dpi=180)
    plt.close(fig)

    print(df.to_string(index=False))


if __name__ == "__main__":
    main()

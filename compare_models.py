"""Compare the three models using primary harmful-content metrics."""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCORES = "results/model_scores.csv"


def main():
    if not os.path.exists(SCORES):
        print("No scores found. Train the models first.")
        return
    df = pd.read_csv(SCORES)
    primary_cols = [
        "model", "harmful_precision", "harmful_recall", "harmful_f1",
        "harmful_false_positive_rate", "harmful_false_negative_rate",
        "harmful_tn", "harmful_fp", "harmful_fn", "harmful_tp",
        "accuracy", "f1_micro", "f1_macro", "train_time_sec",
        "predict_time_sec", "threshold_abusive"
    ]
    table = df[[c for c in primary_cols if c in df.columns]].copy()
    table.to_csv("results/comparison_table.csv", index=False)

    print("\nPRIMARY HARMFUL-CONTENT COMPARISON")
    print(table.to_string(index=False))

    metrics = [
        ("harmful_precision", "Precision"),
        ("harmful_recall", "Recall"),
        ("harmful_f1", "F1"),
        ("harmful_false_positive_rate", "False-positive rate"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (col, title) in zip(axes.ravel(), metrics):
        if col not in table.columns:
            ax.axis("off")
            continue
        ax.bar(table["model"], table[col])
        ax.set_title(title)
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle("Primary harmful-content model comparison")
    fig.tight_layout()
    fig.savefig("results/comparison_f1.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    best = table.loc[table["harmful_f1"].idxmax(), "model"]
    print(f"\nPreferred model by primary harmful-content F1: {best}")


if __name__ == "__main__":
    main()

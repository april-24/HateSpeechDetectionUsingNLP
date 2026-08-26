"""Compare the three models using final-test metrics plus development-only OOF selection."""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCORES = os.path.join("results", "model_scores.csv")


def main():
    if not os.path.exists(SCORES):
        print("No scores found. Train the models first.")
        return
    df = pd.read_csv(SCORES)
    cols = ["model", "accuracy", "subset_accuracy", "hamming_loss",
            "f1_micro", "f1_macro", "f1_weighted",
            "precision_micro", "recall_micro",
            "primary_oof_f1", "primary_test_precision",
            "primary_test_recall", "primary_test_f1",
            "primary_threshold", "label_thresholds",
            "train_time_sec", "predict_time_sec"]
    df = df[[c for c in cols if c in df.columns]]
    print("\n================ MODEL COMPARISON ================\n")
    print(df.to_string(index=False))
    df.to_csv("results/comparison_table.csv", index=False)

    plt.figure(figsize=(9, 5))
    x = range(len(df))
    width = 0.35
    plt.bar([i - width/2 for i in x], df["primary_test_f1"], width,
            label="Final-test abusive F1")
    plt.bar([i + width/2 for i in x], df["primary_oof_f1"], width,
            label="OOF abusive F1")
    plt.xticks(list(x), df["model"], rotation=15, ha="right")
    plt.ylabel("F1 score")
    plt.title("Primary harmful-content performance")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/comparison_primary_f1.png", dpi=120)
    plt.close()

    best = df.loc[df["primary_oof_f1"].idxmax(), "model"]
    print(f"\nDefault model selected from development OOF abusive F1: {best}")
    print("The final test set is not used to choose the default model.")
    print("Saved comparison_table.csv and comparison_primary_f1.png")


if __name__ == "__main__":
    main()

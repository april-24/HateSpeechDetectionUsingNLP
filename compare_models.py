"""Create primary model-comparison charts from saved final-test evidence.

No training or prediction occurs in this script.
"""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import RESULTS_DIR, SCORES_CSV

MODELS = ["Logistic Regression", "Linear SVM", "Random Forest"]


def _save(fig, filename):
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    path = Path(SCORES_CSV)
    if not path.exists():
        raise FileNotFoundError(path)
    scores = pd.read_csv(path)
    order = {m: i for i, m in enumerate(MODELS)}
    scores["_order"] = scores["model"].map(order)
    scores = scores.sort_values("_order").drop(columns="_order")
    cols = [
        "model", "harmful_precision", "harmful_recall", "harmful_f1",
        "harmful_false_positive_rate", "harmful_false_negative_rate",
        "harmful_tn", "harmful_fp", "harmful_fn", "harmful_tp",
        "accuracy", "subset_accuracy", "hamming_loss", "precision_micro",
        "recall_micro", "f1_micro", "precision_macro", "recall_macro", "f1_macro",
        "precision_weighted", "recall_weighted", "f1_weighted", "train_time_sec",
        "predict_time_sec", "threshold_abusive", "oof_primary_f1", "oof_primary_precision",
        "oof_primary_recall", "oof_primary_fpr",
    ]
    table = scores[[c for c in cols if c in scores.columns]].copy()
    table.to_csv(RESULTS_DIR / "comparison_table.csv", index=False)

    x = list(range(len(table)))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (col, label) in enumerate([
        ("harmful_precision", "Precision"),
        ("harmful_recall", "Recall"),
        ("harmful_f1", "F1"),
    ]):
        ax.bar([v + (i-1)*width for v in x], table[col], width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(table["model"], rotation=15)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Primary harmful-content metric comparison")
    ax.legend()
    _save(fig, "primary_metrics_comparison.png")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar([v-width/2 for v in x], table["harmful_false_positive_rate"], width, label="False-positive rate")
    ax.bar([v+width/2 for v in x], table["harmful_false_negative_rate"], width, label="False-negative rate")
    ax.set_xticks(x)
    ax.set_xticklabels(table["model"], rotation=15)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Rate")
    ax.set_title("Primary error-rate comparison")
    ax.legend()
    _save(fig, "primary_error_rates.png")

    print("PRIMARY HARMFUL-CONTENT COMPARISON")
    print(table[["model", "harmful_precision", "harmful_recall", "harmful_f1", "harmful_false_positive_rate", "harmful_false_negative_rate", "f1_micro", "threshold_abusive"]].to_string(index=False))


if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    main()

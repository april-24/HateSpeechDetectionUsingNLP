"""Generate requested final-result charts from saved evidence files.

This module deliberately does not train models or rerun final-test inference.
"""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import RESULTS_DIR

MODELS = ["Logistic Regression", "Linear SVM", "Random Forest"]
LABELS = ["abusive", "Race", "Religion", "Gender", "Sexual_Orientation", "Miscellaneous"]
DISPLAY = {
    "abusive": "Harmful Content", "Race": "Race", "Religion": "Religion",
    "Gender": "Gender", "Sexual_Orientation": "Sexual Orientation", "Miscellaneous": "Miscellaneous",
}


def _save(fig, filename):
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _slug(model):
    return model.lower().replace(" ", "_")


def _load_per_label():
    frames = []
    for model in MODELS:
        path = RESULTS_DIR / f"per_label_{_slug(model)}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        data = pd.read_csv(path).copy()
        data["model"] = model
        frames.append(data)
    return pd.concat(frames, ignore_index=True)


def _load_thresholds():
    frames = []
    for model in MODELS:
        path = RESULTS_DIR / f"threshold_evidence_{_slug(model)}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        data = pd.read_csv(path)[["label", "threshold"]].copy()
        data["model"] = model
        frames.append(data)
    return pd.concat(frames, ignore_index=True)


def _heatmap(data, metric, filename, title):
    piv = data.pivot(index="model", columns="label", values=metric).reindex(index=MODELS, columns=LABELS)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    im = ax.imshow(piv.values, vmin=0, vmax=1, cmap="Blues")
    fig.colorbar(im, ax=ax, label="Score")
    ax.set_xticks(range(len(LABELS)))
    ax.set_xticklabels([DISPLAY[x] for x in LABELS], rotation=30, ha="right")
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels(MODELS)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            value = piv.iloc[i, j]
            ax.text(j, i, f"{value:.3f}" if pd.notna(value) else "—", ha="center", va="center", fontsize=8)
    ax.set_title(title)
    _save(fig, filename)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    scores_path = RESULTS_DIR / "model_scores.csv"
    if not scores_path.exists():
        raise FileNotFoundError(scores_path)
    scores = pd.read_csv(scores_path).set_index("model").reindex(MODELS)
    per_label = _load_per_label()
    thresholds = _load_thresholds()

    for metric, filename, title in [
        ("f1", "per_label_f1_heatmap.png", "Per-label F1 comparison"),
        ("precision", "per_label_precision_heatmap.png", "Per-label precision comparison"),
        ("recall", "per_label_recall_heatmap.png", "Per-label recall comparison"),
    ]:
        _heatmap(per_label, metric, filename, title)

    threshold_piv = thresholds.pivot(index="model", columns="label", values="threshold").reindex(index=MODELS, columns=LABELS)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = range(len(LABELS)); width = 0.25
    for i, model in enumerate(MODELS):
        ax.bar([v + (i-1)*width for v in x], threshold_piv.loc[model].values, width, label=model)
    ax.set_xticks(list(x)); ax.set_xticklabels([DISPLAY[x] for x in LABELS], rotation=30, ha="right")
    ax.set_ylim(0, 1); ax.set_ylabel("Selected threshold")
    ax.set_title("Five-fold OOF selected thresholds"); ax.legend()
    _save(fig, "threshold_comparison.png")

    fig, ax = plt.subplots(figsize=(9, 5))
    x = range(len(MODELS)); width = 0.35
    ax.bar([v-width/2 for v in x], scores["oof_primary_f1"], width, label="OOF F1 (development)")
    ax.bar([v+width/2 for v in x], scores["harmful_f1"], width, label="Final-test F1")
    ax.set_xticks(list(x)); ax.set_xticklabels(MODELS, rotation=15)
    ax.set_ylim(0, 1); ax.set_ylabel("Harmful-content F1")
    ax.set_title("OOF versus final-test harmful-content F1"); ax.legend()
    _save(fig, "oof_vs_final_test_f1.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].bar(MODELS, scores["train_time_sec"]); axes[0].set_title("Training time"); axes[0].set_ylabel("Seconds"); axes[0].tick_params(axis="x", rotation=15)
    axes[1].bar(MODELS, scores["predict_time_sec"]); axes[1].set_title("Final-test prediction time"); axes[1].set_ylabel("Seconds"); axes[1].tick_params(axis="x", rotation=15)
    fig.suptitle("Model timing comparison")
    _save(fig, "model_timing_comparison.png")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, model in zip(axes, MODELS):
        row = scores.loc[model]
        cm = [[int(row.harmful_tn), int(row.harmful_fp)], [int(row.harmful_fn), int(row.harmful_tp)]]
        ax.imshow(cm, cmap="Blues", vmin=0)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i][j], ha="center", va="center", fontsize=11)
        ax.set_xticks([0, 1], ["Pred 0", "Pred 1"]); ax.set_yticks([0, 1], ["Actual 0", "Actual 1"])
        ax.set_title(f"{model}\nthreshold={row.threshold_abusive:.2f}")
    fig.suptitle("Primary harmful-content confusion matrices (final test)")
    _save(fig, "primary_confusion_matrices.png")

    multi = per_label.set_index(["model", "label"]).reindex(pd.MultiIndex.from_product([MODELS, LABELS], names=["model", "label"]))
    fig, axes = plt.subplots(len(MODELS), len(LABELS), figsize=(17, 9))
    for r, model in enumerate(MODELS):
        for c, label in enumerate(LABELS):
            ax = axes[r, c]; row = multi.loc[(model, label)]
            cm = [[int(row.tn), int(row.fp)], [int(row.fn), int(row.tp)]]
            ax.imshow(cm, cmap="Blues", vmin=0)
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, cm[i][j], ha="center", va="center", fontsize=7)
            ax.set_xticks([0, 1]); ax.set_yticks([0, 1]); ax.tick_params(labelsize=7)
            if r == 0: ax.set_title(DISPLAY[label], fontsize=9)
            if c == 0: ax.set_ylabel(model, fontsize=9)
    fig.supxlabel("Predicted"); fig.supylabel("Actual"); fig.suptitle("Six-label confusion matrices (final test)")
    _save(fig, "confusion_matrices_multilabel_final_test.png")
    print("Generated all requested result charts from saved evidence.")


if __name__ == "__main__":
    main()

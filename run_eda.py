"""Generate the complete dataset EDA for HarmShield."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.common import prepare_data
from src.config import DATA_DIR, RESULTS_DIR
from src.data_loader import LABEL_COLS, build_labels, find_csv, load_dataset
from src.preprocessing import preprocess_series

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _save(fig, filename):
    path = RESULTS_DIR / filename
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _pretty(label):
    return label.replace("_", " ")


def main():
    # Use the same dataset-selection function as model training for the model labels
    # and development/test split. This keeps EDA aligned when combined_dataset.csv exists.
    selected_path = Path(find_csv(DATA_DIR))
    model_df, text_col, label_cols = load_dataset(DATA_DIR, verbose=False)
    model_cleaned = preprocess_series(model_df[text_col]).reset_index(drop=True)
    model_labels = model_df[label_cols].reset_index(drop=True)

    # Original-class EDA intentionally uses the immutable HateXplain source. A merged
    # combined dataset contains only adapted binary labels, so it cannot represent the
    # original normal/offensive/hatespeech categories faithfully.
    original_path = DATA_DIR / "final_hateXplain.csv"
    if not original_path.exists():
        raise FileNotFoundError(original_path)
    raw = pd.read_csv(original_path)
    if "comment" not in raw.columns or "label" not in raw.columns:
        raise ValueError("final_hateXplain.csv must contain 'comment' and 'label'")
    raw_nonmissing = raw.dropna(subset=["comment"]).copy()
    raw_nonmissing["comment"] = raw_nonmissing["comment"].astype(str)
    original_labels = build_labels(raw_nonmissing).reset_index(drop=True)
    original_cleaned = preprocess_series(raw_nonmissing["comment"]).reset_index(drop=True)
    lengths = raw_nonmissing["comment"].str.split().apply(len)

    labels = model_labels
    cleaned = model_cleaned

    # 1. Label counts from the dataset selected by the training loader.
    counts = labels[LABEL_COLS].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar([_pretty(x) for x in counts.index], counts.values)
    ax.set_title("Label counts")
    ax.set_ylabel("Number of comments")
    ax.tick_params(axis="x", rotation=25)
    _save(fig, "eda_label_counts.png")

    # 2. Text-length distribution.
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(lengths, bins=50)
    ax.set_title("Text-length distribution")
    ax.set_xlabel("Words per comment")
    ax.set_ylabel("Frequency")
    ax.set_xlim(0, max(1, lengths.quantile(0.99)))
    _save(fig, "eda_text_length.png")

    # 3. Label correlation.
    corr = labels[LABEL_COLS].corr()
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_xticks(range(len(LABEL_COLS)))
    ax.set_yticks(range(len(LABEL_COLS)))
    ax.set_xticklabels([_pretty(x) for x in LABEL_COLS], rotation=35, ha="right")
    ax.set_yticklabels([_pretty(x) for x in LABEL_COLS])
    for i in range(len(LABEL_COLS)):
        for j in range(len(LABEL_COLS)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Label correlation")
    _save(fig, "eda_label_correlation.png")

    # 4. Original HateXplain class distribution (immutable source).
    class_counts = raw_nonmissing["label"].astype(str).str.lower().value_counts().reindex(
        ["normal", "offensive", "hatespeech"], fill_value=0)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(class_counts.index, class_counts.values)
    ax.set_title("Original HateXplain class distribution")
    ax.set_xlabel("Original class")
    ax.set_ylabel("Number of comments")
    _save(fig, "eda_original_class_distribution.png")

    # 5. Positive-label count per comment: all six outputs (training-loader dataset).
    positive_counts = labels[LABEL_COLS].sum(axis=1).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(positive_counts.index.astype(str), positive_counts.values)
    ax.set_title("Positive-label count per comment (six outputs)")
    ax.set_xlabel("Number of positive outputs")
    ax.set_ylabel("Number of comments")
    _save(fig, "eda_positive_labels_per_comment.png")

    # 6. Target-community count per comment: five secondary outputs (training-loader dataset).
    target_cols = LABEL_COLS[1:]
    target_counts = labels[target_cols].sum(axis=1).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(target_counts.index.astype(str), target_counts.values)
    ax.set_title("Target-community count per comment (five target labels)")
    ax.set_xlabel("Number of positive target-community labels")
    ax.set_ylabel("Number of comments")
    _save(fig, "eda_target_count_distribution.png")

    # 7. Target labels by abusive status (training-loader dataset).
    grouped = labels.groupby("abusive")[target_cols].mean().T * 100
    fig, ax = plt.subplots(figsize=(9, 5))
    x = range(len(target_cols))
    width = 0.38
    ax.bar([i-width/2 for i in x], grouped[0], width, label="Non-abusive")
    ax.bar([i+width/2 for i in x], grouped[1], width, label="Abusive")
    ax.set_xticks(list(x))
    ax.set_xticklabels([_pretty(x) for x in target_cols], rotation=25, ha="right")
    ax.set_ylabel("Prevalence (%)")
    ax.set_title("Target labels by abusive status")
    ax.legend()
    _save(fig, "eda_targets_by_abusive_status.png")

    # 8. Text length by original HateXplain class.
    class_order = ["normal", "offensive", "hatespeech"]
    raw_classes = raw_nonmissing["label"].astype(str).str.lower()
    groups = [lengths[raw_classes.eq(c)].to_numpy() for c in class_order]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.boxplot(groups, tick_labels=class_order, showfliers=False)
    ax.set_title("Text length by original class")
    ax.set_ylabel("Words per comment")
    _save(fig, "eda_length_by_original_class.png")

    # 9. Development/test label prevalence.
    X_dev, X_test, y_dev, y_test, _ = prepare_data(DATA_DIR, verbose=False)
    dev_prev = y_dev[LABEL_COLS].mean() * 100
    test_prev = y_test[LABEL_COLS].mean() * 100
    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(LABEL_COLS))
    ax.plot(list(x), dev_prev.values, marker="o", label="Development (80%)")
    ax.plot(list(x), test_prev.values, marker="o", label="Final test (20%)")
    ax.set_xticks(list(x))
    ax.set_xticklabels([_pretty(x) for x in LABEL_COLS], rotation=30, ha="right")
    ax.set_ylabel("Positive prevalence (%)")
    ax.set_title("Development versus final-test label distribution")
    ax.legend()
    _save(fig, "eda_split_distribution.png")

    # 10. Quality summary.
    selected_raw = pd.read_csv(selected_path)
    quality = pd.DataFrame([{
        "eda_training_dataset": selected_path.name,
        "original_hatexplain_record_count": len(raw),
        "selected_dataset_raw_record_count": len(selected_raw),
        "selected_dataset_missing_comments": int(selected_raw["comment"].isna().sum()) if "comment" in selected_raw.columns else -1,
        "selected_dataset_duplicate_comments": int(selected_raw.dropna(subset=["comment"])["comment"].astype(str).duplicated().sum()) if "comment" in selected_raw.columns else -1,
        "records_empty_after_cleaning": int(cleaned.str.len().eq(0).sum()),
        "usable_record_count": int(cleaned.str.len().gt(0).sum()),
        "development_count": int(len(X_dev)),
        "final_test_count": int(len(X_test)),
    }])
    quality.to_csv(RESULTS_DIR / "eda_dataset_quality.csv", index=False)
    print(quality.to_string(index=False))


if __name__ == "__main__":
    main()

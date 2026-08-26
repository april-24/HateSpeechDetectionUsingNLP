"""
run_eda.py
----------
Exploratory Data Analysis. Produces charts you can paste into the
'Methodology / Dataset' section of your documentation.

    python run_eda.py

Outputs (saved to results/):
    eda_label_counts.png     bar chart of how many comments carry each label
    eda_text_length.png      histogram of comment lengths
    eda_label_correlation.png heatmap of how labels co-occur
"""

import os
import matplotlib
matplotlib.use("Agg")            # save figures without a display
import matplotlib.pyplot as plt

from src.data_loader import load_dataset

os.makedirs("results", exist_ok=True)


def main():
    df, text_col, labels = load_dataset("data")

    # 1) label counts
    counts = df[labels].sum().sort_values(ascending=False)
    plt.figure(figsize=(8, 5))
    counts.plot(kind="bar", color="#c0392b")
    plt.title("Number of comments per harmful-content and target label")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("results/eda_label_counts.png", dpi=120)
    plt.close()

    # 2) text length distribution
    lengths = df[text_col].astype(str).str.split().apply(len)
    plt.figure(figsize=(8, 5))
    plt.hist(lengths, bins=50, color="#2980b9")
    plt.title("Comment length (words)")
    plt.xlabel("Words per comment")
    plt.ylabel("Frequency")
    plt.xlim(0, lengths.quantile(0.99))
    plt.tight_layout()
    plt.savefig("results/eda_text_length.png", dpi=120)
    plt.close()

    # 3) label co-occurrence heatmap
    corr = df[labels].corr()
    plt.figure(figsize=(7, 6))
    im = plt.imshow(corr, cmap="Reds", vmin=0, vmax=1)
    plt.colorbar(im)
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center",
                     fontsize=8)
    plt.title("Label co-occurrence (correlation)")
    plt.tight_layout()
    plt.savefig("results/eda_label_correlation.png", dpi=120)
    plt.close()

    # Text summary
    total = len(df)
    clean = (df[labels].sum(axis=1) == 0).sum()
    print(f"\nTotal comments      : {total:,}")
    print(f"Clean (no label)    : {clean:,} ({clean/total:.1%})")
    print(f"Harmful content     : {total-clean:,} ({(total-clean)/total:.1%})")
    print("\nSaved 3 charts to results/ for your documentation.")


if __name__ == "__main__":
    main()

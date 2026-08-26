"""
merge_datasets.py
==================
Combines the original HateXplain dataset with your newly crawled + labeled
Malaysian comments into one training file: data/combined_dataset.csv

After running this, retrain the models as usual - src/data_loader.py
automatically prefers combined_dataset.csv over the original file (see
find_csv() in src/data_loader.py), so no other code changes are needed.

Run from the project root:
    python crawler/merge_datasets.py
"""

import os
import sys
import glob
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_dataset, LABEL_COLS

CRAWLED_DIR = "data/crawled"
OUTPUT_PATH = "data/combined_dataset.csv"


def find_labeled_files():
    return sorted(glob.glob(os.path.join(CRAWLED_DIR, "*_labeled.csv")))


def main():
    print("Loading original dataset...")
    original_df, text_col, labels = load_dataset("data", verbose=False)
    print(f"  {len(original_df):,} comments")

    labeled_files = find_labeled_files()
    if not labeled_files:
        print(f"\nNo labeled files found in {CRAWLED_DIR}/ (looking for *_labeled.csv).")
        print("Run a crawler, then `streamlit run crawler/annotate_data.py` to label "
              "some comments first.")
        return

    print(f"\nFound {len(labeled_files)} labeled file(s):")
    new_frames = []
    for f in labeled_files:
        df = pd.read_csv(f)
        missing = [c for c in LABEL_COLS if c not in df.columns]
        if missing or "comment" not in df.columns:
            print(f"  SKIPPED {f} - missing columns: {missing or ['comment']}")
            continue
        df = df[["comment"] + LABEL_COLS].copy()
        print(f"  {f}: {len(df):,} comments")
        new_frames.append(df)

    if not new_frames:
        print("No valid labeled files to merge.")
        return

    new_df = pd.concat(new_frames, ignore_index=True)

    combined = pd.concat([original_df, new_df], ignore_index=True)
    before = len(combined)
    combined = combined.drop_duplicates(subset=["comment"]).reset_index(drop=True)
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

    dupes_removed = before - len(combined)

    os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
    combined.to_csv(OUTPUT_PATH, index=False)

    print(f"\n{'='*50}")
    print("MERGE COMPLETE")
    print(f"{'='*50}")
    print(f"Original comments      : {len(original_df):,}")
    print(f"New labeled comments    : {len(new_df):,}")
    print(f"Duplicates removed      : {dupes_removed:,}")
    print(f"Final combined dataset  : {len(combined):,}  -> {OUTPUT_PATH}")
    print("\nLabel counts in the combined dataset:")
    print(combined[LABEL_COLS].sum().to_string())
    print("\nNext step: retrain the models (they'll automatically pick up "
          "combined_dataset.csv):")
    print("  python -m models.member1_logistic_regression")
    print("  python -m models.member2_svm")
    print("  python -m models.member3_random_forest")


if __name__ == "__main__":
    main()

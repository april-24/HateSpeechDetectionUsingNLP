"""
data_loader.py
--------------
Loads the HateXplain hate/offensive content dataset (final_hateXplain.csv) and builds the
MULTI-LABEL target matrix.

Dataset columns (final_hateXplain.csv):
    comment                -> the raw text
    label                  -> normal | offensive | hatespeech   (3 classes)
    Race                   -> No_race | African | Arab | ...
    Religion               -> Nonreligious | Islam | Jewish | ...
    Gender                 -> No_gender | Women | Men
    Sexual Orientation     -> No_orientation | Homosexual | ...
    Miscellaneous          -> NaN | Other | Refugee | ...

How we turn this into MULTI-LABEL:
    A single comment can be abusive AND aimed at several communities at once.
    We create SIX binary (0/1) labels per comment:

        abusive              = 1 if label is 'offensive' or 'hatespeech'   (else 0)
        Race                 = 1 if a race community is targeted            (else 0)
        Religion             = 1 if a religion is targeted                  (else 0)
        Gender               = 1 if a gender is targeted                    (else 0)
        Sexual_Orientation   = 1 if a sexual orientation is targeted        (else 0)
        Miscellaneous        = 1 if any 'other' group is targeted           (else 0)

    So the model answers: "Does this comment match the adapted abusive/hate-offensive labels, and who is it aimed at?"
    Because a comment can carry several of these at once, this is a genuine
    multi-label classification problem (5,560 comments target 2+ categories).
"""

import os
import glob
import pandas as pd

TEXT_COL = "comment"

# The 5 target columns and the value that means "not targeted"
TARGET_COLUMNS = {
    "Race": "No_race",
    "Religion": "Nonreligious",
    "Gender": "No_gender",
    "Sexual Orientation": "No_orientation",
    "Miscellaneous": None,          # None means: 'not targeted' == NaN / empty
}

# Final label names (in the binary matrix)
LABEL_COLS = ["abusive", "Race", "Religion", "Gender",
              "Sexual_Orientation", "Miscellaneous"]


def find_csv(data_dir: str) -> str:
    """
    Locate the dataset CSV inside data_dir, searched recursively.
    Priority:
      1. combined_dataset.csv   - the merged HateXplain + your crawled/labeled
                                   data, if you've run crawler/merge_datasets.py
      2. final_hateXplain.csv   - the original Kaggle dataset
      3. anything else with a 'comment' column
    """
    csvs = glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True)
    if not csvs:
        raise FileNotFoundError(
            f"No .csv found under '{data_dir}'. Unzip the Kaggle dataset there first."
        )
    for c in csvs:
        if os.path.basename(c).lower() == "combined_dataset.csv":
            return c
    # Prefer the aggregated 'final_hateXplain.csv' (has the target columns).
    for c in csvs:
        if os.path.basename(c).lower() == "final_hatexplain.csv":
            return c
    # Otherwise prefer any file that actually has a 'comment' column.
    for c in csvs:
        try:
            if "comment" in [x.lower() for x in pd.read_csv(c, nrows=1).columns]:
                return c
        except Exception:
            continue
    # Fallback: biggest file.
    return max(csvs, key=os.path.getsize)


def build_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Create the 6 binary label columns from the raw HateXplain columns."""
    out = pd.DataFrame(index=df.index)

    # 1) 'abusive' from the 3-class label column
    out["abusive"] = df["label"].astype(str).str.lower().isin(
        ["offensive", "hatespeech"]).astype(int)

    # 2) the 5 target-community flags
    for raw_col, none_val in TARGET_COLUMNS.items():
        clean_name = raw_col.replace(" ", "_")
        if none_val is None:
            # 'Miscellaneous': targeted when the cell is NOT empty
            out[clean_name] = df[raw_col].notna().astype(int)
        else:
            out[clean_name] = (df[raw_col].astype(str) != none_val).astype(int)

    return out[LABEL_COLS]


def _is_already_binary(df: pd.DataFrame) -> bool:
    """True if the file already has the 6 final binary label columns (e.g. it
    came from crawler/merge_datasets.py) rather than the raw HateXplain columns."""
    return all(col in df.columns for col in LABEL_COLS)


def load_dataset(data_dir: str = "data", verbose: bool = True):
    """
    Returns (df, text_col, label_cols) where df has the text column and the 6
    binary label columns; label_cols == LABEL_COLS.

    Accepts two input formats automatically:
      - raw HateXplain columns (label, Race, Religion, ...)          -> converted via build_labels()
      - already-binary columns (abusive, Race, Religion, ... as 0/1) -> used as-is
    """
    path = find_csv(data_dir)
    raw = pd.read_csv(path)

    if "comment" not in raw.columns:
        raise ValueError(
            f"'comment' column not found in {path}. Columns: {list(raw.columns)}. "
            "Make sure you are using final_hateXplain.csv or a properly merged dataset."
        )

    # Drop empty comments
    raw = raw.dropna(subset=[TEXT_COL]).reset_index(drop=True)
    raw[TEXT_COL] = raw[TEXT_COL].astype(str)

    if _is_already_binary(raw):
        labels = raw[LABEL_COLS].fillna(0).astype(int)
    else:
        labels = build_labels(raw)
    df = pd.concat([raw[[TEXT_COL]], labels], axis=1)

    if verbose:
        print(f"Loaded file   : {path}")
        print(f"Rows          : {len(df):,}")
        print(f"Text column   : '{TEXT_COL}'")
        print(f"Label columns : {LABEL_COLS}")
        print("\nPositive count per label:")
        print(df[LABEL_COLS].sum().to_string())
        n_active = df[LABEL_COLS].sum(axis=1)
        print(f"\nComments with 2+ labels (multi-label): {(n_active >= 2).sum():,}")
        print("\nSample row:")
        print(df.iloc[0].to_string())

    return df, TEXT_COL, LABEL_COLS


if __name__ == "__main__":
    load_dataset("data")

"""Shared data preparation for a leakage-safe model comparison.

The final 20% test set is created once and is never used for model selection
or threshold selection.  Five-fold cross-validation is subsequently performed
inside the remaining 80% development set by :mod:`src.train_utils`.
"""

from sklearn.model_selection import train_test_split
from .data_loader import load_dataset
from .config import PROJECT_ROOT, DATA_DIR
from .preprocessing import preprocess_series

RANDOM_STATE = 42
TEST_SIZE = 0.2


def stratification_key(y, min_count=2):
    """Return a reproducible, reasonably balanced label-combination key.

    Common six-label signatures are kept intact. Rare signatures are grouped
    by the primary harmful-content label and number of target groups. This
    avoids failing on very rare combinations while preserving the most
    important multilabel distribution characteristics without an extra
    dependency.
    """
    signatures = y.astype(str).agg("".join, axis=1)
    counts = signatures.value_counts()
    rare = signatures.map(counts) < min_count
    target_count = y.iloc[:, 1:].sum(axis=1).clip(upper=3).astype(str)
    fallback = "primary_" + y.iloc[:, 0].astype(str) + "_targets_" + target_count
    key = signatures.where(~rare, fallback)
    if key.value_counts().min() < min_count:
        broader = fallback
        if broader.value_counts().min() >= min_count:
            return broader
        return "primary_" + y.iloc[:, 0].astype(str)
    return key


def prepare_data(data_dir=DATA_DIR, sample=None, verbose=True):
    """
    Returns:
        X_dev, X_test : cleaned text (pandas Series)
        y_dev, y_test : label matrices (DataFrames of 0/1)
        label_cols      : list of label names
    `sample` (int) optionally sub-samples the data for a quick test run.
    """
    df, text_col, label_cols = load_dataset(data_dir, verbose=verbose)

    if sample:
        df = df.sample(n=min(sample, len(df)), random_state=RANDOM_STATE).reset_index(drop=True)
        print(f"[info] Using a random sample of {len(df):,} rows for a quick run.")

    if verbose:
        print("\nCleaning text ... (this can take a minute on the full dataset)")
    X = preprocess_series(df[text_col])
    y = df[label_cols]

    # Drop rows that became empty after cleaning
    mask = X.str.len() > 0
    X, y = X[mask].reset_index(drop=True), y[mask].reset_index(drop=True)

    split_key = stratification_key(y, min_count=2)
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE,
        stratify=split_key)

    if verbose:
        print(f"Development size: {len(X_dev):,}   Final test size: {len(X_test):,}")
        print("The final test set must not be used for tuning or threshold selection.")
    return X_dev, X_test, y_dev, y_test, label_cols

"""Shared reproducible 80/20 train/final-test split for all models."""
from sklearn.model_selection import train_test_split
from .data_loader import load_dataset
from .preprocessing import preprocess_series

RANDOM_STATE = 42
TEST_SIZE = 0.20


def prepare_data(data_dir="data", sample=None, verbose=True):
    """Return an untouched final test set plus the 80% development set.

    Returns:
        X_train, X_test, y_train, y_test, label_cols

    Threshold and model selection must be performed only through cross-validation
    inside X_train/y_train. The final 20% test split is never used for tuning.
    """
    df, text_col, label_cols = load_dataset(data_dir, verbose=verbose)
    if sample:
        df = df.sample(n=min(sample, len(df)), random_state=RANDOM_STATE).reset_index(drop=True)
        if verbose:
            print(f"[info] Using a random sample of {len(df):,} rows for a quick run.")

    if verbose:
        print("\nCleaning text ... (this can take a minute on the full dataset)")
    X = preprocess_series(df[text_col])
    y = df[label_cols]
    mask = X.str.len() > 0
    X, y = X[mask].reset_index(drop=True), y[mask].reset_index(drop=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    if verbose:
        print(f"Train/development size: {len(X_train):,}   Final test size: {len(X_test):,}")
    return X_train, X_test, y_train, y_test, label_cols

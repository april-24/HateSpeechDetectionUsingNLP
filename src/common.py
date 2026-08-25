"""
common.py
---------
Shared reproducible data split for all three models.

The data are divided into training, validation, and final test sets:
60% training, 20% validation, 20% final test. Model fitting and TF-IDF
vocabulary learning use only the training portion. The validation set is
used for model/threshold decisions, while the final test set remains
untouched until the final evaluation.
"""
from sklearn.model_selection import train_test_split
from .data_loader import load_dataset
from .preprocessing import preprocess_series

RANDOM_STATE = 42
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.25  # 25% of the 80% development set = 20% overall

def prepare_data(data_dir="data", sample=None, verbose=True):
    """
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test, label_cols
    """
    df, text_col, label_cols = load_dataset(data_dir, verbose=verbose)

    if sample:
        df = df.sample(n=min(sample, len(df)), random_state=RANDOM_STATE).reset_index(drop=True)
        print(f"[info] Using a random sample of {len(df):,} rows for a quick run.")

    if verbose:
        print("\nCleaning text ... (this can take a minute on the full dataset)")
    X = preprocess_series(df[text_col])
    y = df[label_cols]

    mask = X.str.len() > 0
    X, y = X[mask].reset_index(drop=True), y[mask].reset_index(drop=True)

    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_dev, y_dev, test_size=VALIDATION_SIZE, random_state=RANDOM_STATE)

    if verbose:
        print(f"Train size: {len(X_train):,}   Validation size: {len(X_val):,}   Final test size: {len(X_test):,}")
    return X_train, X_val, X_test, y_train, y_val, y_test, label_cols

"""
threshold_sweep.py
------------------
Recomputes model-specific decision thresholds using the validation split only.
The final test set is never used for threshold selection.
"""
from .common import prepare_data
from .predictor import available_models, load_model
from .train_utils import select_threshold

def sweep():
    Xtr, Xval, Xte, ytr, yval, yte, labels = prepare_data(verbose=False)
    results = {}
    for name, path in available_models().items():
        bundle = load_model(path)
        th, val_f1 = select_threshold(bundle["pipeline"], Xval, yval)
        results[name] = th
        print(f"{name:22s} threshold={th:.2f} validation_micro-F1={val_f1:.4f}")
    return results

if __name__ == "__main__":
    sweep()

"""
threshold_sweep.py
-------------------
Displays thresholds selected from five-fold pooled out-of-fold predictions
during training. It never re-selects thresholds on the final test set.

Why this matters: different algorithms produce probability-like scores on
different natural scales, even when equally correct. Forcing every model to
share one threshold (e.g. 0.60) can badly under-serve some of them - on this
project's models, Random Forest's F1 dropped from 0.70 (at its own best
threshold) to 0.56 when forced to share a 0.60 cutoff with the others.

Run this any time you retrain a model - the right threshold is specific to
that exact trained model, not to the algorithm in general, so it can shift
if you retrain (e.g. after merging in more data).

Run from the project root:
    python -m src.threshold_sweep
"""

from .predictor import available_models, load_model


def sweep():
    print(f"{'Model':22s} {'OOF threshold':15s} {'Selection source'}")
    results = {}
    for name, path in available_models().items():
        bundle = load_model(path)
        threshold = float(bundle.get("threshold", 0.5))
        source = bundle.get("threshold_source", "unknown")
        results[name] = threshold
        print(f"{name:22s} {threshold:<15.2f} {source}")

    print("\nCopy this into DEFAULT_THRESHOLDS in src/config.py:")
    print("DEFAULT_THRESHOLDS = {")
    for name, th in results.items():
        print(f'    "{name}": {th},')
    print("}")
    return results


if __name__ == "__main__":
    sweep()

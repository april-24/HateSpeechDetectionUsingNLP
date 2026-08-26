"""Display the six saved OOF thresholds for each model."""

from .config import LABELS
from .predictor import available_models, load_model


def sweep():
    print(f"{'Model':22s} " + " ".join(f"{l:19s}" for l in LABELS))
    results = {}
    for name, path in available_models().items():
        bundle = load_model(path)
        thresholds = bundle.get("thresholds", {})
        if not thresholds:
            thresholds = {l: bundle.get("threshold", 0.5) for l in LABELS}
        results[name] = thresholds
        print(f"{name:22s} " + " ".join(f"{float(thresholds.get(l, 0.5)):.2f}{' '*16}" for l in LABELS))
        print("  source:", bundle.get("threshold_source", "unknown"))

    print("\nThresholds are selected from pooled 3-fold OOF predictions on the development set.")
    print("The final test set is not used for threshold selection.")
    return results


if __name__ == "__main__":
    sweep()

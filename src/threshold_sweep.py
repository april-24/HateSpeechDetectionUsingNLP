"""Show the six thresholds saved from development-only OOF selection."""
from .predictor import available_models, load_model
from .config import LABELS


def sweep():
    results = {}
    print(f"{'Model':22s} " + " ".join(f"{l:20s}" for l in LABELS))
    for name, path in available_models().items():
        bundle = load_model(path)
        thresholds = bundle["thresholds"]
        results[name] = thresholds
        print(f"{name:22s} " + " ".join(
            f"{float(thresholds[l]):<20.2f}" for l in LABELS))
    print("\nSaved OOF thresholds by model:")
    for name, thresholds in results.items():
        print(name, thresholds)
    return results


if __name__ == "__main__":
    sweep()

"""Recompute model-specific thresholds using CV on the 80% training data only."""
import joblib
from .common import prepare_data
from .train_utils import select_threshold_cv

MODELS = {
    "Logistic Regression": "results/model_lr.joblib",
    "Linear SVM": "results/model_svm.joblib",
    "Random Forest": "results/model_rf.joblib",
}


def main():
    X_train, X_test, y_train, y_test, labels = prepare_data(verbose=False)
    for name, path in MODELS.items():
        bundle = joblib.load(path)
        th, f1 = select_threshold_cv(bundle["pipeline"], X_train, y_train)
        print(f"{name:22s} CV threshold={th:.2f} OOF_micro-F1={f1:.4f}")


if __name__ == "__main__":
    main()

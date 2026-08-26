"""Central configuration for the hate/offensive content detector.

All project paths are resolved from this file so the app works whether the
project directory is the repository root or a nested GitHub subfolder.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
DATA_DIR = PROJECT_ROOT / "data"

LABELS = ["abusive", "Race", "Religion", "Gender",
          "Sexual_Orientation", "Miscellaneous"]
PRIMARY_LABEL = "abusive"
PRIMARY_DECISION_DESCRIPTION = (
    "Only the abusive label determines the final harmful-content YES/NO verdict."
)

DISPLAY_NAMES = {
    "abusive": "Harmful Content",
    "Race": "Race Target",
    "Religion": "Religion Target",
    "Gender": "Gender Target",
    "Sexual_Orientation": "Sexual-Orientation Target",
    "Miscellaneous": "Other Community Target",
}

MODEL_FILES = {
    "Logistic Regression": str(RESULTS_DIR / "model_lr.joblib"),
    "Linear SVM": str(RESULTS_DIR / "model_svm.joblib"),
    "Random Forest": str(RESULTS_DIR / "model_rf.joblib"),
}

# Six-label fallback thresholds. Retrained model bundles override these values.
DEFAULT_THRESHOLDS = {
    "Logistic Regression": {
        "abusive": 0.50, "Race": 0.50, "Religion": 0.50,
        "Gender": 0.50, "Sexual_Orientation": 0.50, "Miscellaneous": 0.50,
    },
    "Linear SVM": {
        "abusive": 0.50, "Race": 0.50, "Religion": 0.50,
        "Gender": 0.50, "Sexual_Orientation": 0.50, "Miscellaneous": 0.50,
    },
    "Random Forest": {
        "abusive": 0.50, "Race": 0.50, "Religion": 0.50,
        "Gender": 0.50, "Sexual_Orientation": 0.50, "Miscellaneous": 0.50,
    },
}

SCORES_CSV = str(RESULTS_DIR / "model_scores.csv")


def pretty(label: str) -> str:
    return DISPLAY_NAMES.get(label, label)


def get_default_thresholds(model_name):
    return dict(DEFAULT_THRESHOLDS.get(model_name, {
        label: 0.50 for label in LABELS
    }))

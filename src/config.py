"""Central configuration for the hate/offensive content detector."""

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
    "Logistic Regression": "results/model_lr.joblib",
    "Linear SVM": "results/model_svm.joblib",
    "Random Forest": "results/model_rf.joblib",
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

SCORES_CSV = "results/model_scores.csv"
DATA_DIR = "data"


def pretty(label: str) -> str:
    return DISPLAY_NAMES.get(label, label)


def get_default_thresholds(model_name):
    return dict(DEFAULT_THRESHOLDS.get(model_name, {
        label: 0.50 for label in LABELS
    }))

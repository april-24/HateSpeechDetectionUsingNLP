"""Central configuration for the HarmShield Hate/Offensive Content Detector."""

LABELS = ["abusive", "Race", "Religion", "Gender",
          "Sexual_Orientation", "Miscellaneous"]
PRIMARY_LABEL = "abusive"

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

# Fallbacks are used only when a model bundle is missing threshold metadata.
# They are per-label, not one scalar shared by every output.
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

# Threshold-selection objective. Keep F1 as the default so the choice is
# transparent and balanced. Set to "f_beta" and choose beta > 1 when recall
# is deliberately prioritised by the project/report.
PRIMARY_THRESHOLD_METRIC = "f1"
PRIMARY_F_BETA = 1.5

# Optional UI status. A prediction below the primary threshold but close to it,
# with at least one positive target-group output, is surfaced for human review.
BORDERLINE_REVIEW_ENABLED = True
BORDERLINE_MARGIN = 0.08

SCORES_CSV = "results/model_scores.csv"
DATA_DIR = "data"


def pretty(label: str) -> str:
    return DISPLAY_NAMES.get(label, label)

"""
config.py
---------
Central place for label names, human-friendly category names, and file paths,
so every part of the system (training, evaluation, Streamlit app) stays consistent.
"""

# The 6 binary labels produced by src/data_loader.py (order matters).
LABELS = ["abusive", "Race", "Religion", "Gender",
          "Sexual_Orientation", "Miscellaneous"]
PRIMARY_LABEL = "abusive"

# Friendly names shown in the user interface. These map directly onto the labels
# the HateXplain dataset actually provides — we do NOT invent categories the data
# was never labelled for (e.g. 'profanity', 'threat' are not separate labels here).
DISPLAY_NAMES = {
    "abusive":            "Harmful Content",
    "Race":               "Race Target",
    "Religion":           "Religion Target",
    "Gender":             "Gender Target",
    "Sexual_Orientation": "Sexual-Orientation Target",
    "Miscellaneous":      "Other Community Target",
}

# Model registry: display name -> saved file in results/
MODEL_FILES = {
    "Logistic Regression": "results/model_lr.joblib",
    "Linear SVM":          "results/model_svm.joblib",
    "Random Forest":       "results/model_rf.joblib",
}

# These fallbacks are overwritten by thresholds saved inside newly trained
# model bundles. They are selected from pooled out-of-fold predictions on the
# 80% development set, never from the final test set.
DEFAULT_THRESHOLDS = {
    "Logistic Regression": 0.27,
    "Linear SVM": 0.46,
    "Random Forest": 0.42,
}

SCORES_CSV = "results/model_scores.csv"
DATA_DIR = "data"


def pretty(label: str) -> str:
    """Return the friendly display name for a raw label."""
    return DISPLAY_NAMES.get(label, label)

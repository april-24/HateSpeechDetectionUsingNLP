"""
config.py
---------
Central place for label names, human-friendly category names, and file paths,
so every part of the system (training, evaluation, Streamlit app) stays consistent.
"""

# The 6 binary labels produced by src/data_loader.py (order matters).
LABELS = ["abusive", "Race", "Religion", "Gender",
          "Sexual_Orientation", "Miscellaneous"]

# Friendly names shown in the user interface. These map directly onto the labels
# the HateXplain dataset actually provides — we do NOT invent categories the data
# was never labelled for (e.g. 'profanity', 'threat' are not separate labels here).
DISPLAY_NAMES = {
    "abusive":            "Abusive / Harmful Content",
    "Race":               "Race Target",
    "Religion":           "Religion Target",
    "Gender":             "Gender Target",
    "Sexual_Orientation": "Sexual Orientation Target",
    "Miscellaneous":      "Other Target",
}

# Model registry: display name -> saved file in results/
MODEL_FILES = {
    "Logistic Regression": "results/model_lr.joblib",
    "Linear SVM":          "results/model_svm.joblib",
    "Random Forest":       "results/model_rf.joblib",
}

# Per-model default detection thresholds. NOT arbitrary - each was found by
# selecting thresholds on the validation set and then freezing them before final test evaluation
# that maximizes micro-F1 for THAT model (verified against the exact trained
# models currently in results/ - re-run this sweep any time a model is
# retrained, since the right threshold shifts if the model itself changes).
#
# Why per-model at all: a single shared threshold (e.g. 0.60) implicitly
# assumes all models' probability outputs mean the same thing, but they
# don't. Random Forest's predict_proba (ensemble vote fraction) sits on a
# different natural scale than Logistic Regression's directly-fitted
# probability, even when equally correct. Sharing one threshold penalizes
# some models far more than others; measured impact of forcing a shared 0.60
# on the models currently shipped in results/:
#   Logistic Regression : validation F1 0.7051 at threshold 0.44
#   Linear SVM           : validation F1 0.6869 at threshold 0.48
#   Random Forest        : validation F1 0.7001 at threshold 0.46
DEFAULT_THRESHOLDS = {
    # Updated automatically after leakage-safe validation calibration.
    "Logistic Regression": 0.44,
    "Linear SVM": 0.48,
    "Random Forest": 0.46,
}

SCORES_CSV = "results/model_scores.csv"
DATA_DIR = "data"


def pretty(label: str) -> str:
    """Return the friendly display name for a raw label."""
    return DISPLAY_NAMES.get(label, label)

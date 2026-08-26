"""Automated regression/behavioural evaluation for the final detector."""
import os
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)

from .config import MODEL_FILES, PRIMARY_LABEL, PROJECT_ROOT, RESULTS_DIR
from .predictor import load_model, _label_probs
from .preprocessing import clean_text


def evaluate_model_cases(model_name, path, cases):
    bundle = load_model(path)
    texts = cases["text"].astype(str).tolist()
    probs = _label_probs(bundle["pipeline"], [clean_text(t) for t in texts])
    idx = bundle["labels"].index(PRIMARY_LABEL)
    threshold = float(bundle["thresholds"][PRIMARY_LABEL])
    pred = (probs[:, idx] >= threshold).astype(int)
    expected = cases["expected_harmful"].astype(int).to_numpy()

    rows = cases.copy()
    rows["model"] = model_name
    rows["harmful_probability"] = probs[:, idx]
    rows["abusive_threshold"] = threshold
    rows["predicted_harmful"] = pred
    rows["predicted_output"] = np.where(pred == 1, "YES", "NO")
    rows["correct"] = pred == expected
    return rows


def main():
    cases_path = str(PROJECT_ROOT / "data" / "behavioral_test_cases.csv")
    if not os.path.exists(cases_path):
        raise FileNotFoundError(cases_path)
    cases = pd.read_csv(cases_path)
    all_details = []
    for name, path in MODEL_FILES.items():
        if os.path.exists(path):
            all_details.append(evaluate_model_cases(name, path, cases))
    if not all_details:
        raise FileNotFoundError("No trained model files found.")
    details = pd.concat(all_details, ignore_index=True)

    summaries = []
    for (model, category), g in details.groupby(["model", "category"]):
        y = g.expected_harmful.astype(int)
        p = g.predicted_harmful.astype(int)
        summaries.append({
            "model": model,
            "category": category,
            "n": len(g),
            "accuracy": accuracy_score(y, p),
            "harmful_recall": recall_score(
                y, p, zero_division=0) if y.sum() else np.nan,
            "harmful_f1": f1_score(y, p, zero_division=0) if y.sum() else np.nan,
            "benign_false_positive_rate": (
                ((p == 1) & (y == 0)).sum() / max((y == 0).sum(), 1)
            ),
        })
    category_df = pd.DataFrame(summaries)

    overall=[]
    for model,g in details.groupby("model"):
        y=g.expected_harmful.astype(int); p=g.predicted_harmful.astype(int)
        tn,fp,fn,tp=confusion_matrix(y,p,labels=[0,1]).ravel()
        overall.append({
            "model":model,
            "accuracy":accuracy_score(y,p),
            "harmful_precision":precision_score(y,p,zero_division=0),
            "harmful_recall":recall_score(y,p,zero_division=0),
            "harmful_f1":f1_score(y,p,zero_division=0),
            "benign_false_positive_rate":fp/max(fp+tn,1),
            "tn":tn,"fp":fp,"fn":fn,"tp":tp,
        })
    overall_df=pd.DataFrame(overall)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    details.to_csv(str(RESULTS_DIR / "behavioral_test_details.csv"),index=False)
    category_df.to_csv(str(RESULTS_DIR / "behavioral_category_results.csv"),index=False)
    overall_df.to_csv(str(RESULTS_DIR / "behavioral_comparison.csv"),index=False)

    print("\nBehavioural evaluation")
    print(overall_df.to_string(index=False))
    print("\nCategory-level results")
    print(category_df.to_string(index=False))


if __name__ == "__main__":
    main()

import os, re, joblib, numpy as np, pandas as pd
from sklearn.metrics import f1_score
from src.common import prepare_data
from src.config import MODEL_FILES, LABELS, PRIMARY_LABEL, PRIMARY_THRESHOLD_METRIC, PRIMARY_F_BETA
from src.evaluate import evaluate_model, save_result
from src.predictor import _label_probs
from src.train_utils import choose_thresholds, apply_thresholds, _save_per_label


def finalize(name, path):
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    oof = pd.read_csv(f"results/oof_predictions_{slug}.csv")
    prob = oof[[f"prob_{l}" for l in LABELS]].to_numpy()
    y = oof[[f"true_{l}" for l in LABELS]].to_numpy()
    thresholds, metrics = choose_thresholds(y, prob, LABELS)
    print("THRESHOLDS", name, thresholds)
    bundle = joblib.load(path)
    X_dev, X_test, y_dev, y_test, labels = prepare_data(verbose=False)
    P = _label_probs(bundle["pipeline"], list(X_test))
    pred = apply_thresholds(P, thresholds, labels)
    result = evaluate_model(name, y_test.values, pred, labels, probabilities=P, thresholds=thresholds)
    result.update({
        "primary_threshold": thresholds[PRIMARY_LABEL],
        "threshold_selection_metric": PRIMARY_THRESHOLD_METRIC,
        "primary_f_beta": PRIMARY_F_BETA if PRIMARY_THRESHOLD_METRIC == "f_beta" else "",
        "thresholds": "; ".join(f"{l}={thresholds[l]:.2f}" for l in labels),
        "oof_f1_micro": round(float(f1_score(y, apply_thresholds(prob, thresholds, labels), average="micro", zero_division=0)),4),
    })
    save_result(result)
    _save_per_label(name, y_test.values, pred, labels, thresholds)
    bundle.update({
        "labels": labels,
        "thresholds": thresholds,
        "threshold": thresholds[PRIMARY_LABEL],
        "primary_threshold": thresholds[PRIMARY_LABEL],
        "primary_threshold_metric": PRIMARY_THRESHOLD_METRIC,
        "primary_threshold_beta": PRIMARY_F_BETA,
        "threshold_metrics_oof": metrics,
        "threshold_source": "development OOF predictions; final 20% test set not used for threshold selection",
        "oof_micro_f1": result["oof_f1_micro"],
        "oof_predictions_file": f"results/oof_predictions_{slug}.csv",
    })
    joblib.dump(bundle, path, compress=3)

if __name__ == '__main__':
    import sys
    finalize(sys.argv[1], sys.argv[2])

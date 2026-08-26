"""Generate final-test confusion matrices and behavioural/evasion artifacts."""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from src.common import prepare_data
from src.config import MODEL_FILES, PRIMARY_LABEL, PROJECT_ROOT, RESULTS_DIR
from src.predictor import load_model, _label_probs
from src.preprocessing import clean_text
from src.behavioral_evaluation import main as run_behavioral


def primary_confusion_figure():
    _, X_test, _, y_test, labels = prepare_data(verbose=False)
    primary_index = labels.index(PRIMARY_LABEL)
    fig, axes = plt.subplots(1, len(MODEL_FILES), figsize=(12, 4))
    if len(MODEL_FILES) == 1:
        axes = [axes]
    for ax, (model_name, path) in zip(axes, MODEL_FILES.items()):
        if not os.path.exists(path):
            ax.axis("off")
            continue
        bundle = load_model(path)
        P = _label_probs(bundle["pipeline"], list(X_test))
        threshold = float(bundle["thresholds"][PRIMARY_LABEL])
        pred = (P[:, primary_index] >= threshold).astype(int)
        cm = confusion_matrix(
            y_test.values[:, primary_index], pred, labels=[0, 1])
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=11)
        ax.set_title(f"{model_name}\nthreshold={threshold:.2f}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    fig.suptitle("Final-test primary harmful-content confusion matrices")
    fig.tight_layout()
    fig.savefig(
        str(RESULTS_DIR / "confusion_matrices_final_test.png"),
        dpi=180, bbox_inches="tight")
    plt.close(fig)


def multilabel_confusion_figure():
    _, X_test, _, y_test, labels = prepare_data(verbose=False)
    fig, axes = plt.subplots(
        len(MODEL_FILES), len(labels), figsize=(17, 9))
    for row, (model_name, path) in enumerate(MODEL_FILES.items()):
        if not os.path.exists(path):
            continue
        bundle = load_model(path)
        P = _label_probs(bundle["pipeline"], list(X_test))
        for col, label in enumerate(labels):
            th = float(bundle["thresholds"][label])
            pred = (P[:, col] >= th).astype(int)
            cm = confusion_matrix(
                y_test.values[:, col], pred, labels=[0, 1])
            ax = axes[row, col]
            ax.imshow(cm, cmap="Blues")
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8)
            ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
            if row == 0:
                ax.set_title(label, fontsize=9)
            if col == 0:
                ax.set_ylabel(model_name, fontsize=9)
    fig.supxlabel("Predicted")
    fig.supylabel("Actual")
    fig.tight_layout()
    fig.savefig(str(RESULTS_DIR / "confusion_matrices_multilabel_final_test.png"),
                dpi=170, bbox_inches="tight")
    plt.close(fig)


def evasion_demonstration():
    """Keep the evasion result explicitly qualitative; behavioural evaluation
    is the quantitative larger test."""
    cases = pd.read_csv(PROJECT_ROOT / "data" / "behavioral_test_cases.csv")
    cases = cases[cases["category"] == "obfuscated_harmful"].copy()
    rows=[]
    for model_name,path in MODEL_FILES.items():
        if not os.path.exists(path):
            continue
        bundle=load_model(path)
        cleaned=[clean_text(x) for x in cases["text"]]
        P=_label_probs(bundle["pipeline"],cleaned)
        idx=bundle["labels"].index(PRIMARY_LABEL)
        th=float(bundle["thresholds"][PRIMARY_LABEL])
        for (_,case),prob in zip(cases.iterrows(),P[:,idx]):
            rows.append({
                "model":model_name,
                "case_type":"obfuscated_harmful",
                "text":case["text"],
                "threshold":th,
                "harmful_probability":float(prob),
                "predicted_harmful":int(prob>=th),
                "expected_harmful":int(case["expected_harmful"])
            })
    details=pd.DataFrame(rows)
    summary=details.groupby("model").agg(
        detected=("predicted_harmful","sum"),
        total=("predicted_harmful","count")
    ).reset_index()
    summary["detection_rate"]=summary["detected"]/summary["total"]
    summary["interpretation"]="Qualitative demonstration; not a claim of adversarial robustness."
    details.to_csv(str(RESULTS_DIR / "evasion_demonstration_details.csv"),index=False)
    summary.to_csv(str(RESULTS_DIR / "evasion_demonstration_summary.csv"),index=False)


if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    primary_confusion_figure()
    multilabel_confusion_figure()
    if (PROJECT_ROOT / "data" / "behavioral_test_cases.csv").exists():
        run_behavioral()
        evasion_demonstration()
    print("Generated final-test confusion matrices and behavioural/evasion artifacts.")

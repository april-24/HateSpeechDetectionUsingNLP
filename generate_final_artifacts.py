"""Generate final-test figures and a transparent expanded evasion demonstration."""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from src.common import prepare_data
from src.config import MODEL_FILES, pretty, PRIMARY_LABEL
from src.predictor import load_model, _label_probs
from src.preprocessing import clean_text


def confusion_figure():
    _, X_test, _, y_test, labels = prepare_data(verbose=False)
    fig, axes = plt.subplots(len(MODEL_FILES), len(labels), figsize=(16, 8))
    for row, (model_name, path) in enumerate(MODEL_FILES.items()):
        bundle = load_model(path)
        threshold = float(bundle["threshold"])
        probabilities = _label_probs(bundle["pipeline"], list(X_test))
        predictions = (probabilities >= threshold).astype(int)
        for col, label in enumerate(labels):
            matrix = confusion_matrix(y_test.values[:, col], predictions[:, col])
            ax = axes[row, col]
            ax.imshow(matrix, cmap="Blues")
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, matrix[i, j], ha="center", va="center", fontsize=8)
            if row == 0:
                ax.set_title(pretty(label), fontsize=9)
            if col == 0:
                ax.set_ylabel(model_name, fontsize=9)
            ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    fig.supxlabel("Predicted label")
    fig.supylabel("Actual label")
    fig.tight_layout()
    fig.savefig("results/confusion_matrices_final_test.png", dpi=180)
    plt.close(fig)


def evasion_demonstration():
    cases = [
        ("harmful_plain", "you are a stupid idiot"),
        ("harmful_obfuscated", "you are a st00pid i.d.i.o.t"),
        ("harmful_plain", "I hate you and you are worthless"),
        ("harmful_obfuscated", "I h4te you and you are w0rthless"),
        ("harmful_plain", "go back to your own country"),
        ("harmful_obfuscated", "g.o b4ck to your own c0untry"),
        ("harmful_plain", "women are too stupid to lead"),
        ("harmful_obfuscated", "w0men are too stuuuupid to lead"),
        ("harmful_plain", "gay people are disgusting"),
        ("harmful_obfuscated", "g4y people are d.i.s.g.u.s.t.i.n.g"),
        ("harmful_plain", "you pathetic loser"),
        ("harmful_obfuscated", "you p4thetic l0ser"),
        ("benign", "thank you for your helpful explanation"),
        ("benign", "I disagree with this policy"),
        ("benign", "women participated in the leadership meeting"),
        ("benign", "the Christian community held a charity event"),
        ("benign", "Muslim students organised a food donation drive"),
        ("benign", "the race begins at nine in the morning"),
    ]
    rows = []
    cleaned = [clean_text(text) for _, text in cases]
    for model_name, path in MODEL_FILES.items():
        bundle = load_model(path)
        threshold = float(bundle["threshold"])
        probabilities = _label_probs(bundle["pipeline"], cleaned)
        primary_index = bundle["labels"].index(PRIMARY_LABEL)
        for (case_type, text), clean, probability in zip(
                cases, cleaned, probabilities[:, primary_index]):
            rows.append({
                "model": model_name,
                "case_type": case_type,
                "text": text,
                "cleaned_text": clean,
                "threshold": threshold,
                "harmful_probability": float(probability),
                "predicted_harmful": int(probability >= threshold),
            })
    details = pd.DataFrame(rows)
    details.to_csv("results/evasion_demonstration_details.csv", index=False)
    summary = details.groupby(["model", "case_type"])["predicted_harmful"].agg(
        detected="sum", total="count", rate="mean").reset_index()
    summary.to_csv("results/evasion_demonstration_summary.csv", index=False)


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    confusion_figure()
    evasion_demonstration()
    print("Generated final-test confusion matrices and evasion demonstration artifacts.")

"""Generate final-test confusion matrices and a larger behavioural evasion demo."""
import os
import re
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from src.common import prepare_data
from src.config import MODEL_FILES, pretty, PRIMARY_LABEL
from src.predictor import load_model, _label_probs, get_thresholds
from src.train_utils import apply_thresholds
from src.preprocessing import clean_text


CASES = [
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
    ("harmful_plain", "immigrants do not belong here"),
    ("harmful_obfuscated", "immigr4nts d0 n0t bel0ng here"),
    ("harmful_plain", "Muslim people are disgusting"),
    ("harmful_plain", "men are useless"),
    ("harmful_plain", "stop insulting everyone"),
    ("harmful_plain", "you are an idiot"),
    ("benign", "thank you for your helpful explanation"),
    ("benign", "Thanks for sharing, this was really useful!"),
    ("benign", "I disagree with this policy"),
    ("benign", "women participated in the leadership meeting"),
    ("benign", "the Christian community held a charity event"),
    ("benign", "Muslim students organised a food donation drive"),
    ("benign", "the race begins at nine in the morning"),
    ("benign", "I am not happy with the result but I respect your view"),
    ("benign", "Could you please explain your point again?"),
    ("benign", "This discussion is useful and informative"),
]


def confusion_figure():
    _, X_test, _, y_test, labels = prepare_data(verbose=False)
    n_models = len(MODEL_FILES)
    fig, axes = plt.subplots(n_models, len(labels), figsize=(16, 8))
    if n_models == 1:
        axes = [axes]
    for row, (model_name, path) in enumerate(MODEL_FILES.items()):
        bundle = load_model(path)
        thresholds = get_thresholds(bundle, model_name=model_name)
        probabilities = _label_probs(bundle["pipeline"], list(X_test))
        predictions = apply_thresholds(probabilities, thresholds, labels)
        for col, label in enumerate(labels):
            matrix = confusion_matrix(y_test.values[:, col], predictions[:, col])
            ax = axes[row][col]
            ax.imshow(matrix)
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
    rows = []
    cleaned = [clean_text(text) for _, text in CASES]
    for model_name, path in MODEL_FILES.items():
        bundle = load_model(path)
        thresholds = get_thresholds(bundle, model_name=model_name)
        probabilities = _label_probs(bundle["pipeline"], cleaned)
        predictions = apply_thresholds(probabilities, thresholds, bundle["labels"])
        primary_index = bundle["labels"].index(PRIMARY_LABEL)
        for idx, ((case_type, text), clean) in enumerate(zip(CASES, cleaned)):
            primary_prob = float(probabilities[idx, primary_index])
            primary_pred = int(predictions[idx, primary_index])
            target_positive = [
                bundle["labels"][j] for j in range(1, len(bundle["labels"]))
                if predictions[idx, j] == 1
            ]
            rows.append({
                "model": model_name,
                "case_type": case_type,
                "text": text,
                "cleaned_text": clean,
                "primary_threshold": float(thresholds[PRIMARY_LABEL]),
                "harmful_probability": primary_prob,
                "predicted_harmful": primary_pred,
                "positive_target_labels": ", ".join(target_positive),
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
    print("Generated final-test confusion matrices and expanded behavioural evaluation.")

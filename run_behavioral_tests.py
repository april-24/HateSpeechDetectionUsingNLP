"""Development-only behavioural sanity test.

This set is NOT used for training, threshold selection, model selection, or final
reporting. It is a qualitative development check for obvious harmful text, implicit
identity attacks, hard negatives, benign identity discussions, and obfuscation.
"""
import os
import pandas as pd
from src.config import MODEL_FILES, PRIMARY_LABEL
from src.predictor import load_model, predict


def main():
    cases = pd.read_csv("data/development_behavioral_tests.csv")
    rows = []
    for model_name, path in MODEL_FILES.items():
        if not os.path.exists(path):
            continue
        bundle = load_model(path)
        for _, row in cases.iterrows():
            result = predict(bundle, row["text"])
            rows.append({
                "model": model_name,
                "case_type": row["case_type"],
                "expected_harmful": int(row["expected_harmful"]),
                "predicted_harmful": int(result["is_harmful"]),
                "primary_probability": result["probs"][PRIMARY_LABEL],
                "primary_threshold": result["threshold"],
                "text": row["text"],
            })
    out = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    out.to_csv("results/development_behavioral_test_results.csv", index=False)
    summary = out.groupby(["model", "case_type"]).agg(
        detected=("predicted_harmful", "sum"),
        total=("predicted_harmful", "count"),
        rate=("predicted_harmful", "mean"),
        expected_rate=("expected_harmful", "mean"),
    ).reset_index()
    summary.to_csv("results/development_behavioral_test_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

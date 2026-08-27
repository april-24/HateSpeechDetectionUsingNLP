# HarmShield — Hate and Offensive Content Detection

HarmShield is an educational NLP prototype for BMCS2074 Artificial Intelligence. It detects **hate and offensive content** in individual English posts and predicts five target-community annotations for context.

## Scope and decision rule

HateXplain contains individual posts, so the project is described as **hate and offensive content detection**, not complete cyberbullying detection. It does not claim to identify repeated behaviour, power imbalance, victim–aggressor relationships, or conversation-level cyberbullying.

The six binary outputs are `abusive`, `Race`, `Religion`, `Gender`, `Sexual_Orientation`, and `Miscellaneous`. Only `abusive` controls the final verdict:

```text
abusive probability >= saved abusive threshold -> YES
otherwise                                      -> NO
```

Target-community predictions are contextual and can never turn a primary `NO` into `YES`.

## Leakage-safe methodology

The existing methodology is preserved: reserve 20% as the untouched final-test set; use the complete 80% development partition for five-fold model-configuration CV; fit TF-IDF inside fold training data; run a separate five-fold OOF procedure on the full development partition for thresholds; fit the selected model on all development data; then evaluate once on the untouched final test set.

Expected cleaned split: **16,086 development + 4,022 final-test records**. Saved primary thresholds are LR **0.37**, calibrated Linear SVM **0.49**, and RF **0.47**.

## Models

| Model | Representation | Classifier |
|---|---|---|
| Logistic Regression | Word TF-IDF 1–3 + character TF-IDF 3–5 | One-vs-Rest LR, balanced class weight |
| Calibrated Linear SVM | Word TF-IDF 1–3 + character TF-IDF 3–5 | One-vs-Rest LinearSVC with sigmoid calibration |
| Random Forest | Word TF-IDF | One-vs-Rest Random Forest, balanced class weight |

## Final-test evidence

| Model | Abusive threshold | Harmful precision | Harmful recall | Harmful F1 | Benign FPR | Multilabel micro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| **Logistic Regression** | **0.37** | 0.7597 | **0.8524** | **0.8034** | 0.4242 | **0.7245** |
| Linear SVM | 0.49 | **0.7628** | 0.8451 | 0.8019 | 0.4133 | 0.7126 |
| Random Forest | 0.47 | 0.7729 | 0.7654 | 0.7691 | **0.3538** | 0.7104 |

Logistic Regression remains the preferred/default model based mainly on final harmful-content F1 and multilabel micro-F1.

## Regression Example Testing

Only the three required regression examples remain as a separate prediction check. They do not belong to the training data and do not influence model selection, threshold selection, or final-test evaluation. Predictions use the real preprocessing, saved model bundle, and saved threshold.

| Text | LR | SVM | RF |
|---|---|---|---|
| `you are a stupid idiot nobody likes you` | YES | YES | YES |
| `Thanks for sharing, this was really useful!` | NO | NO | NO |
| `go back to your own country you don't belong here` | YES | YES | YES |

All three exact sentences are absent from `data/final_hateXplain.csv`, so they are not training rows. They are not hard-coded into `predict()`.

The removed supplementary evaluation component and its generated evidence are not included in the final project. This keeps the final evaluation focused on the saved development/test methodology and the three required regression examples.

## Revised files and commands

Key files:

```text
app.py                              Streamlit prototype; no automatic retraining
run_eda.py                          complete dataset EDA + quality CSV
compare_models.py                  primary metrics/error-rate charts + comparison table
generate_final_artifacts.py        all requested final result charts from saved evidence
Hate_Offensive_Content_Detection.ipynb
                                    technical demonstration and reproducibility record
models/                             LR, calibrated SVM, RF model definitions
src/common.py                       protected 80/20 split
src/train_utils.py                 five-fold model selection + separate five-fold OOF thresholds
src/predictor.py                   saved-threshold prediction engine
results/model_scores.csv           final-test evidence
results/per_label_*.csv            final per-label evidence
results/threshold_evidence_*.csv   OOF threshold evidence
results/oof_*.csv                  OOF probability evidence
```

Install:

```bash
python -m pip install -r requirements.txt
```

EDA:

```bash
python run_eda.py
```

Result charts (no retraining):

```bash
python compare_models.py
python generate_final_artifacts.py
```

Notebook: open `Hate_Offensive_Content_Detection.ipynb`. It normally loads saved results. Full retraining is protected by `RUN_FULL_RETRAINING = False`.

Streamlit:

```bash
streamlit run app.py
```

Explicit training commands, when a new training run is intentionally required:

```bash
python -m models.member1_logistic_regression
python -m models.member2_svm
python -m models.member3_random_forest
```

## EDA outputs

`run_eda.py` generates:

- `eda_original_class_distribution.png`
- `eda_label_counts.png`
- `eda_text_length.png`
- `eda_label_correlation.png`
- `eda_positive_labels_per_comment.png`
- `eda_target_count_distribution.png`
- `eda_targets_by_abusive_status.png`
- `eda_length_by_original_class.png`
- `eda_split_distribution.png`
- `eda_dataset_quality.csv`

The EDA distinguishes all-six-output positive counts from five-label target-community counts and avoids interpreting multiple positive labels as multiple independent harmful behaviours.

## Final result charts

The project now generates:

- `primary_metrics_comparison.png`
- `primary_error_rates.png`
- `per_label_f1_heatmap.png`
- `per_label_precision_heatmap.png`
- `per_label_recall_heatmap.png`
- `threshold_comparison.png`
- `oof_vs_final_test_f1.png`
- `model_timing_comparison.png`
- `primary_confusion_matrices.png`
- `confusion_matrices_multilabel_final_test.png`

These charts read the saved evidence CSVs and do not recalculate different final-test results.

## Notebook purpose

The notebook covers project introduction, setup, dataset loading, quality checks, EDA, preprocessing, TF-IDF, the 80/20 split, five-fold model selection, separate five-fold OOF thresholds, final evaluation, charts, per-label results, confusion matrices, the three regression examples, model comparison, limitations, and conclusion.

## Streamlit behaviour

The app has organised EDA/results tabs and handles missing images gracefully. It loads saved model bundles and results at startup and does not retrain automatically. There is no manual threshold slider.

## Limitations

- HateXplain is not a dedicated cyberbullying dataset; it contains individual posts rather than conversation histories or behaviour trajectories.
- English-only short-form social-media data may not generalise to other languages or domains.
- Target labels can occur in normal posts, and harmful posts can exist without a mapped target community.
- Gender and Miscellaneous are weaker outputs in the supplied final-test evidence.
- Model explanations are feature contributions, not causal explanations.
- Statistical predictions can be wrong; deployment decisions require human judgement.

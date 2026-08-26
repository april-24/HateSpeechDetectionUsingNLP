# HarmShield - Hate and Offensive Content Detection

HarmShield is an educational NLP application that compares Logistic Regression,
calibrated Linear SVM, and Random Forest pipelines on the HateXplain dataset.
The system predicts one primary harmful-content output and five supporting
target-community outputs.

## Important interpretation

- `abusive` is the primary harmful-content output. It is 1 when HateXplain's
  original label is `offensive` or `hatespeech`.
- Race, Religion, Gender, Sexual Orientation, and Miscellaneous are
  target-community annotations.
- A target-community prediction does **not** independently create a harmful
  verdict. The application flags a post only when the primary `abusive`
  probability crosses its own saved threshold.
- This is not a dedicated cyberbullying dataset and the system does not infer
  repeated behaviour, a victim-aggressor relationship, or conversation history.

## Leakage-safe threshold selection

1. A stratified 20% final test set is reserved.
2. Cross-validation runs only on the 80% development set.
3. TF-IDF is fitted inside each development fold, so validation text does not
   leak into the vocabulary or document statistics used to train that fold.
4. Six thresholds are selected independently from pooled OOF probabilities:
   one for `abusive` and one for each target-community label.
5. The `abusive` threshold is selected using its own binary F1 objective; it is
   not selected from six-label micro-F1.
6. The fixed six thresholds are then applied once to the untouched final test
   set. No final-test result is used to choose a threshold.

The saved OOF probability files are included in `results/` so the threshold
selection can be audited without rerunning cross-validation.

LinearSVC has no native probabilities, so the SVM uses
`CalibratedClassifierCV(method="sigmoid", cv=3)`. The application does not create a
fake probability by applying a sigmoid manually to a raw SVM margin.

## Final results

| Model | Primary threshold | Primary F1 | Primary recall | Primary FPR | Multilabel accuracy | Micro-F1 | Macro-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.33 | **0.7993** | 0.8609 | 0.4613 | **0.8258** | **0.6986** | **0.6604** |
| Linear SVM | 0.44 | 0.7964 | 0.8804 | 0.5202 | 0.8200 | 0.6936 | 0.6526 |
| Random Forest | 0.42 | 0.7826 | **0.9105** | 0.6552 | 0.8119 | 0.6733 | 0.6204 |

Logistic Regression is the preferred balanced default because it has the
highest primary harmful-content F1 and the strongest multilabel accuracy and
F1 in the final comparison. Linear SVM has the highest primary recall, while
Random Forest is the most recall-oriented but also produces substantially more
false positives.

## Example content-detection behaviour

Using the saved per-label thresholds, the built-in three-comment example now
produces:

| Comment | Logistic Regression | Linear SVM | Random Forest |
|---|---|---|---|
| `you are a stupid idiot nobody likes you` | YES | YES | YES |
| `Thanks for sharing, this was really useful!` | NO | NO | YES |
| `go back to your own country you don't belong here` | YES | YES | YES |

The results illustrate why threshold calibration improves recall and makes
model operating points explicit. Random Forest still produces a false positive
on the polite example; this is retained as an explicit model limitation rather
than hidden with a hand-coded exception.

## Installation and use

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python compare_models.py
python generate_final_artifacts.py
streamlit run app.py
```

The trained bundles already included in `results/` contain the fitted pipelines,
per-label thresholds, OOF threshold metrics, and references to pooled OOF
prediction files.

## Application pages

- **Home** - project scope, objective, models, and dataset summary.
- **Dataset Statistics** - output prevalence and exploratory figures.
- **Data Preprocessing** - step-by-step cleaning and TF-IDF preview.
- **Content Detection** - individual, batch, CSV, or supported URL analysis.
- **Model Evaluation** - primary harmful-content metrics, multilabel metrics,
  model comparison, and final-test reports.

## Main files

```text
app.py                         Streamlit application
models/                        three member model definitions
src/common.py                  protected holdout split
src/train_utils.py             OOF threshold selection and threshold application
src/predictor.py               prediction and linear-model explanations
src/preprocessing.py           cleaning and evasion normalisation
src/evaluate.py                primary + multilabel evaluation
src/threshold_sweep.py         six saved thresholds per model
results/model_scores.csv      final test and OOF summary
results/comparison_table.csv  primary vs multilabel model comparison
results/per_label_*.csv       final per-output results
results/oof_predictions_*.csv pooled OOF probabilities and decisions
results/evasion_*.csv         expanded behavioural demonstration
```

## Known limitations

- HateXplain concerns hate/offensive speech rather than cyberbullying behaviour.
- English-only data may not generalise to other languages or domains.
- Target-community annotations can occur in posts labelled normal.
- Gender and Miscellaneous remain weaker outputs because of class imbalance and
  label ambiguity.
- The evasion experiment is a behavioural demonstration, not an adversarial
  robustness benchmark.
- Linear explanations show learned feature contributions, not human causes.
- Random Forest explanations are not presented as local word causes because
  global feature importance is not a valid local explanation.
- Automated moderation decisions require human review.

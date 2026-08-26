# HarmShield - Hate and Offensive Content Detection

HarmShield is an educational NLP application that compares Logistic Regression,
calibrated Linear SVM, and Random Forest pipelines on the HateXplain dataset.
The system predicts a primary harmful-content output and five supporting
target-community outputs.

## Important interpretation

- `abusive` is the primary harmful-content output. It is 1 when HateXplain's
  original label is `offensive` or `hatespeech`.
- Race, Religion, Gender, Sexual Orientation, and Miscellaneous are
  target-community annotations.
- A target-community prediction does **not** independently create a harmful
  verdict. The application flags a post only when the primary output crosses
  the selected model's threshold.
- This is not a dedicated cyberbullying dataset and the system does not infer
  repeated behaviour, a victim-aggressor relationship, or conversation history.

## Leakage-safe evaluation

1. A stratified 20% final test set is reserved.
2. Five-fold cross-validation runs only on the 80% development set.
3. TF-IDF is fitted independently inside every fold.
4. One threshold per model is selected from pooled out-of-fold probabilities.
5. The final pipeline is fitted on the complete development set.
6. The selected threshold is applied once to the untouched final test set.

LinearSVC has no native probabilities, so it is wrapped in
`CalibratedClassifierCV(method="sigmoid", cv=2)`. No sigmoid is manually
applied to raw SVM margins.

Each model receives its own out-of-fold-selected threshold for the primary
`abusive` label and separate thresholds for the five target-community labels.
The final application uses a two-stage decision: the primary harmful-content
decision is made first, and target groups are surfaced only when that decision is positive.

## Final results

| Model | OOF abusive threshold | Accuracy | Primary F1 | Micro-F1 | Macro-F1 | Subset accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.27 | **0.8357** | **0.8075** | **0.7138** | **0.6722** | **0.4010** |
| Linear SVM | 0.46 | 0.8348 | 0.8032 | 0.7058 | 0.6624 | 0.4023 |
| Random Forest | 0.42 | 0.8283 | 0.7850 | 0.6984 | 0.6528 | 0.3804 |

Logistic Regression has the highest final-test primary harmful-content F1 and
highest overall micro-F1, while Linear SVM has the highest primary precision.
The default model is therefore selected from development-side OOF abusive F1,
which favours Logistic Regression.

## Installation and use

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m models.member1_logistic_regression
python -m models.member2_svm
python -m models.member3_random_forest
python compare_models.py
python generate_final_artifacts.py
streamlit run app.py
```

The saved bundles already included in `results/` contain the final fitted
pipelines and their out-of-fold-selected thresholds, so retraining is optional
for demonstration.

## Application pages

- **Home** - project scope, objective, models, and dataset summary.
- **Dataset Statistics** - output prevalence and exploratory figures.
- **Data Preprocessing** - step-by-step cleaning and TF-IDF preview.
- **Content Detection** - individual, batch, CSV, or supported URL analysis.
- **Model Evaluation** - final-test metrics, model comparison, and reports.

## Main files

```text
app.py                         Streamlit application
models/                        three member model definitions
src/common.py                  protected holdout split
src/train_utils.py             five-fold OOF selection and final training
src/predictor.py               prediction and local linear explanations
src/preprocessing.py           cleaning and evasion normalisation
results/model_scores.csv       final test and cross-validation summary
results/per_label_*.csv        final per-output results
results/evasion_*.csv          expanded qualitative demonstration
```

## Known limitations

- HateXplain concerns hate/offensive speech rather than cyberbullying behaviour.
- English-only data from Twitter and Gab may not generalise to other domains.
- Target-community annotations can occur in posts labelled normal.
- Gender and Miscellaneous remain difficult outputs.
- The small evasion demonstration is not an adversarial robustness benchmark.
- Linear explanations show model contributions, not causes.
- Random Forest word highlighting is disabled because global feature importance
  is not a valid local explanation.
- Automated moderation decisions require human review.

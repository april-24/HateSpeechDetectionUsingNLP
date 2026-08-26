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
`CalibratedClassifierCV(method="sigmoid", cv=3)`. No sigmoid is manually
applied to raw SVM margins.

## Final results

| Model | OOF threshold | Accuracy | Micro-F1 | Macro-F1 | Subset accuracy |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.48 | 0.8352 | 0.7105 | 0.6893 | 0.3645 |
| Linear SVM | 0.31 | 0.8335 | **0.7179** | 0.6693 | 0.3523 |
| Random Forest | 0.48 | 0.8232 | 0.6930 | 0.6828 | 0.3170 |

Linear SVM has the highest final-test micro-F1 and recall. Logistic Regression
has the highest accuracy, macro-F1, weighted F1, and subset accuracy, making it
the preferred balanced default. Random Forest trains and predicts fastest in
this run but has lower overall effectiveness.

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

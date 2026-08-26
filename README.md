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
2. Development-only cross-validation is used for model configuration selection.
3. TF-IDF is fitted independently inside every training fold.
4. Three-fold OOF probabilities from a fixed 2,000-row subset of the 80% development data are used for threshold selection.
5. The `abusive` threshold is selected using harmful-content F1 with a minimum precision preference of 0.75; each target label gets its own OOF F1 threshold.
6. The final pipeline is fitted on the complete development set.
7. The selected thresholds are applied once to the untouched final test set.

LinearSVC has no native probabilities, so it is wrapped in
`CalibratedClassifierCV(method="sigmoid", cv=3)`. Calibration is fitted only
inside the training-side data used by the classifier; no sigmoid is manually
applied to raw SVM margins.

## Final results

| Model | Abusive threshold | Harmful precision | Harmful recall | Harmful F1 | Benign FPR | Multilabel micro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.53 | 0.8363 | 0.6897 | 0.7560 | 0.2124 | 0.6761 |
| Linear SVM | 0.59 | 0.8237 | 0.7296 | **0.7738** | 0.2457 | 0.6949 |
| Random Forest | 0.50 | **0.8511** | 0.6251 | 0.7208 | **0.1721** | 0.6202 |

The final primary verdict comparison selects **Linear SVM by harmful-content F1**. Random Forest has the highest primary precision and lowest benign false-positive rate, while Logistic Regression provides a strong balanced alternative. These primary metrics are separate from the six-label multilabel metrics.

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
src/train_utils.py             three-fold OOF selection and final training
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


## Final decision rule (revised)

The system is a **hate/offensive content detector**, not a dedicated cyberbullying classifier. The HateXplain annotations are adapted into six binary outputs: `abusive` plus five target-community annotations.

### Fixed primary YES/NO rule

Only the `abusive` label determines the final harmful-content verdict:

- `abusive probability >= saved abusive threshold` -> **YES**
- otherwise -> **NO**

The five target labels (Race, Religion, Gender, Sexual Orientation, Miscellaneous) are contextual outputs. They use their own independently selected thresholds and **cannot turn a NO into YES**.

### Leakage-safe threshold and model selection

The final 20% test set is kept untouched. Model configurations are selected using cross-validation within the 80% development data. Three-fold out-of-fold predictions from the development set are then used to select thresholds. The `abusive` threshold is optimized using harmful-content F1, with precision, recall and benign false-positive rate reported; each target label is optimized independently using its own F1.

The saved model bundle contains a `thresholds` dictionary and an OOF evidence CSV. The Streamlit interface does not expose a manual threshold slider, so the evaluated decision rule cannot be changed accidentally.

### Behavioural evaluation

`data/behavioral_test_cases.csv` contains 48 regression/behavioural cases covering direct insults, implicit attacks, benign compliments, benign identity discussion, quotations, negations, obfuscated harmful text and hate-target examples. The three required regression checks are included:

1. `you are a stupid idiot nobody likes you` -> YES
2. `Thanks for sharing, this was really useful!` -> NO
3. `go back to your own country you don't belong here` -> YES

Run:

```bash
python -m models.member1_logistic_regression
python -m models.member2_svm
python -m models.member3_random_forest
python compare_models.py
python generate_final_artifacts.py
```

The final artifacts distinguish primary harmful-content metrics from the six-label multilabel metrics.

### Final model/threshold notes

The final implementation uses development-only cross-validation for model configuration and a separate three-fold OOF stage on a fixed 2,000-row subset of the 80% development data for threshold selection. The final 20% test set remains untouched for selection.

For the primary `abusive` threshold, harmful-content F1 is the objective with a minimum precision preference of 0.75 to avoid an excessively aggressive moderation operating point. Precision, recall, F1, FPR and FNR are all reported. Each target-community label has its own independent OOF F1 threshold.

The shared TF-IDF representation evaluates word unigrams, bigrams and trigrams. The preprocessing stage also expands common contractions before punctuation removal, preserving negations such as `not` and `don't`/`do not` consistently.

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

1. A stratified 20% final test set is reserved and never used for tuning.
2. The full 80% development partition is used for **five-fold cross-validation** for LR, SVM and RF configuration selection.
3. TF-IDF is fitted inside each training fold, so validation/test vocabulary statistics do not influence training.
4. A separate **five-fold OOF** procedure on the full 80% development data produces probabilities for threshold selection.
5. The `abusive` threshold is selected from the primary label's OOF predictions using harmful-content F1 with a minimum precision preference of 0.75; each target label receives its own OOF F1 threshold.
6. The selected pipeline is fitted on the complete 80% development set.
7. The fixed thresholds are applied once to the untouched 20% final test set.

The final model bundles and result files in this ZIP were regenerated after the five-fold changes using the deterministic preprocessing configuration.

## Final results

| Model | Abusive threshold | Harmful precision | Harmful recall | Harmful F1 | Benign FPR | Multilabel micro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| **Logistic Regression** | **0.37** | 0.7597 | **0.8524** | **0.8034** | 0.4242 | **0.7245** |
| Linear SVM | 0.49 | **0.7628** | 0.8451 | 0.8019 | 0.4133 | 0.7126 |
| Random Forest | 0.47 | 0.7729 | 0.7654 | 0.7691 | **0.3538** | 0.7104 |

Based on the final-test **harmful-content F1**, Logistic Regression is the preferred model in this regenerated five-fold version. Random Forest has the lowest benign false-positive rate, while Logistic Regression has the highest harmful-content recall and F1.

### Regression checks

The required examples produce the following primary harmful-content verdicts for all three models:

| Text | LR | SVM | RF |
|---|---|---|---|
| `you are a stupid idiot nobody likes you` | YES | YES | YES |
| `Thanks for sharing, this was really useful!` | NO | NO | NO |
| `go back to your own country you don't belong here` | YES | YES | YES |

## Reproducibility and deployment

- The project supports Python 3.11 (`runtime.txt`).
- `scikit-learn`, NumPy, pandas, joblib, Streamlit and WordCloud are pinned in `requirements.txt`.
- No NLTK resource is downloaded when the application starts. Text preprocessing uses deterministic built-in rules, which avoids cloud startup delays caused by network downloads.
- All dataset, model and result paths are resolved from `src/config.py` using `Path(__file__).resolve()`. Therefore the project can be placed directly at the GitHub repository root **or inside a subfolder** without breaking paths.
- Model training is not performed when Streamlit starts; the deployed app loads the saved model bundles from `results/`.

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


## Final decision rule (revised)

The system is a **hate/offensive content detector**, not a dedicated cyberbullying classifier. The HateXplain annotations are adapted into six binary outputs: `abusive` plus five target-community annotations.

### Fixed primary YES/NO rule

Only the `abusive` label determines the final harmful-content verdict:

- `abusive probability >= saved abusive threshold` -> **YES**
- otherwise -> **NO**

The five target labels (Race, Religion, Gender, Sexual Orientation, Miscellaneous) are contextual outputs. They use their own independently selected thresholds and **cannot turn a NO into YES**.

### Leakage-safe threshold and model selection

The final 20% test set is kept untouched. Model configurations are selected using cross-validation within the 80% development data. Five-fold out-of-fold predictions from the development set are then used to select thresholds. The `abusive` threshold is optimized using harmful-content F1, with precision, recall and benign false-positive rate reported; each target label is optimized independently using its own F1.

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

The final implementation uses development-only cross-validation for model configuration and a separate five-fold OOF stage on a full 80% development data for threshold selection. The final 20% test set remains untouched for selection.

For the primary `abusive` threshold, harmful-content F1 is the objective with a minimum precision preference of 0.75 to avoid an excessively aggressive moderation operating point. Precision, recall, F1, FPR and FNR are all reported. Each target-community label has its own independent OOF F1 threshold.

The shared TF-IDF representation evaluates word unigrams, bigrams and trigrams. The preprocessing stage also expands common contractions before punctuation removal, preserving negations such as `not` and `don't`/`do not` consistently.

# HarmShield User Guide

## Start the application

Install the packages in `requirements.txt`, then run:

```powershell
streamlit run app.py
```

## Select a model

The application provides Logistic Regression, calibrated Linear SVM, and
Random Forest. When a model is selected, its decision threshold is read from
the saved model bundle. Each threshold was selected using five-fold
out-of-fold predictions on the development set, not the final test set.

There is no sensitivity/threshold slider. The evaluated threshold is fixed so the same decision rule is used during demonstration and batch analysis.

## Interpret an output

- **Harmful-content probability** is the primary moderation output.
- A post is flagged only when this primary probability crosses the threshold.
- Target-community probabilities are supporting outputs. They indicate which
  community the language may concern; they cannot create a harmful verdict by
  themselves.
- Highlighted words are model contributions rather than causes. They are
  available for the linear models. Random Forest highlighting is intentionally
  disabled because its global feature importance is not a local explanation.

## Pages

### Home

Explains the project scope, objectives, dataset, and three fitted pipelines.

### Dataset Statistics

Shows the HateXplain output distribution, text-length distribution, and output
correlations. The primary `abusive` column represents harmful content; the
remaining columns are target-community annotations.

### Data Preprocessing

Shows lowercase conversion, URL/HTML removal, evasion normalisation,
tokenisation, stop-word handling, and lemmatisation. Negations such as `not`,
`no`, `nor`, and `never` are preserved.

### Content Detection

Use one of three tabs:

1. **Enter Comment** - test one comment or several lines.
2. **Import CSV** - select a text column and analyse rows in a file.
3. **Social Media URL** - use supported public integrations where configured.

Always review flagged posts manually before acting.

### Model Evaluation

Shows final-test metrics and side-by-side model predictions. Official metrics
come from the untouched 20% final test set using the threshold saved with each
model.

## Safety and limitations

HarmShield is a course prototype, not an autonomous moderation authority. It
does not understand user relationships, repeated behaviour, sarcasm, cultural
context, or the complete conversation. It can generate false positives and
false negatives. Use it to prioritise human review, not to punish users
automatically.


## Fixed harmful-content decision

The Content Detection page uses the trained model's saved OOF-selected thresholds. There is **no manual threshold slider**.

Only the `abusive` probability determines the final result:

**YES** when `abusive probability >= abusive threshold`; otherwise **NO**.

Race, Religion, Gender, Sexual Orientation and Miscellaneous are displayed as separate target-group context. Their individual thresholds do not independently create a YES result.

The same fixed rule is used for single comments, imported CSV/TXT files and crawled/social-media comments.

The displayed primary probability and primary threshold are separate fields so the decision can be explained and reproduced.

## Regression examples

The final system should return the following primary harmful-content verdicts for all three trained models:

- `you are a stupid idiot nobody likes you` -> **YES**
- `Thanks for sharing, this was really useful!` -> **NO**
- `go back to your own country you don't belong here` -> **YES**

These are regression checks only. They are not used to select the threshold.

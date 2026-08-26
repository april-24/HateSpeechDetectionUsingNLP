# Final Version Notes

This archive is the final revised prototype for the AI/NLP group assignment.
The project terminology is **Hate and Offensive Content Detection** using the
HateXplain dataset. It should not be presented as a dedicated cyberbullying
dataset.

## Main corrections applied

1. The final 20% test set remains untouched for threshold selection.
2. Threshold selection is performed on development-data out-of-fold (OOF)
   probabilities.
3. Six independent thresholds are stored for each model: `abusive`, `Race`,
   `Religion`, `Gender`, `Sexual_Orientation`, and `Miscellaneous`.
4. The primary `abusive` threshold is selected independently using its own
   binary F1 objective. It is no longer derived from six-label micro-F1.
5. The predictor applies the correct threshold column-by-column and uses only
   `abusive` for the main harmful YES/NO verdict.
6. Bulk CSV and supported social-media analysis use the same threshold logic.
7. Model evaluation now reports primary harmful-content accuracy, precision,
   recall, F1, false-positive rate and false-negative rate separately from
   multilabel metrics.
8. Confusion matrices and the model-comparison figure were regenerated using
   the fixed thresholds.
9. The evasion/behavioural demonstration was expanded and is explicitly
   described as a qualitative demonstration, not an adversarial robustness
   benchmark.
10. The Streamlit UI no longer treats a manual threshold slider as the official
    evaluation setting.
11. README, USER_GUIDE, and the training notebook were updated to describe the
    per-label threshold design and the primary harmful-content verdict.

## Final saved primary thresholds

| Model | `abusive` threshold |
|---|---:|
| Logistic Regression | 0.33 |
| Linear SVM | 0.44 |
| Random Forest | 0.42 |

See `results/threshold_audit.csv` and the three `results/oof_predictions_*.csv`
files for the detailed OOF audit trail.

## Example system behaviour

- `you are a stupid idiot nobody likes you` -> LR YES, SVM YES, RF YES
- `Thanks for sharing, this was really useful!` -> LR NO, SVM NO, RF YES
- `go back to your own country you don't belong here` -> LR YES, SVM YES, RF YES

The Random Forest false positive on the polite example is retained as a real
model limitation rather than being hard-coded away.

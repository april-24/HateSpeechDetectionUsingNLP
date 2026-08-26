"""
predictor.py
------------
The prediction engine used by the Streamlit app. It loads a saved model bundle
and, for a given comment, returns:
    - per-label probability / confidence
    - the binary prediction at the model's saved OOF-selected threshold
    - the words that most influenced the decision (for highlighting)
    - a short plain-English explanation

Works for all three models:
    Logistic Regression / Random Forest -> predict_proba
    Linear SVM                           -> calibrated predict_proba
Word influence:
    linear models -> signed coefficients (coef_)
    random forest -> omitted because global importance is not a local explanation
"""

import os
import numpy as np
import joblib

from .preprocessing import clean_text
from .config import (LABELS, PRIMARY_LABEL, pretty, MODEL_FILES, DEFAULT_THRESHOLDS,
                     BORDERLINE_REVIEW_ENABLED, BORDERLINE_MARGIN)


def available_models():
    """Return {display_name: path} for the models that are actually saved."""
    return {name: path for name, path in MODEL_FILES.items() if os.path.exists(path)}


def load_model(path):
    """Load a saved {pipeline, labels} bundle.

    Forces n_jobs=1 on any RandomForest estimators inside the pipeline. RF
    was trained with n_jobs=-1 to use all CPU cores during training (good -
    training processes many rows at once), but for a SINGLE-comment
    prediction that same setting can make things slower, not faster: joblib
    has to spin up a parallel worker pool for one tiny row of work, and on a
    constrained server (e.g. Streamlit Community Cloud's shared/limited CPU)
    that overhead can dominate over the actual computation. Sequential
    (n_jobs=1) evaluation of a single row is typically faster in practice.
    """
    bundle = joblib.load(path)
    try:
        clf = bundle["pipeline"].named_steps.get("clf")
        estimators = getattr(clf, "estimators_", None) or []
        for est in estimators:
            if hasattr(est, "n_jobs"):
                est.n_jobs = 1
    except Exception:
        pass  # non-RF models have no n_jobs attribute - nothing to do
    return bundle


def _label_probs(pipeline, texts):
    """Return calibrated/fitted probabilities; never squash raw margins."""
    if not hasattr(pipeline, "predict_proba"):
        raise TypeError("Saved model has no predicted probabilities; retrain it with calibration.")
    return np.asarray(pipeline.predict_proba(texts))


def get_thresholds(bundle, model_name=None):
    """Return the saved per-label threshold dictionary."""
    thresholds = bundle.get("thresholds")
    if isinstance(thresholds, dict):
        return {label: float(thresholds.get(label, 0.5)) for label in bundle.get("labels", LABELS)}
    # Backward compatibility for old bundles with one scalar threshold.
    labels = bundle.get("labels", LABELS)
    fallback = DEFAULT_THRESHOLDS.get(model_name, {}) if model_name else {}
    scalar = float(bundle.get("threshold", 0.5))
    return {label: float(fallback.get(label, scalar)) for label in labels}


def predict(bundle, text, threshold=None, model_name=None):
    """Analyze one comment using the saved threshold for EACH label.

    ``abusive`` alone controls the primary harmful-content YES/NO verdict.
    Target-community labels are contextual and cannot turn a clean comment
    into a harmful verdict.
    """
    labels = bundle["labels"]
    pipe = bundle["pipeline"]
    cleaned = clean_text(text)

    thresholds = get_thresholds(bundle, model_name=model_name)
    # Optional scalar threshold is retained only as a demonstration override;
    # reported evaluation results always use the saved per-label thresholds.
    if threshold is not None:
        thresholds = {label: float(threshold) for label in labels}
        override_used = True
    else:
        override_used = False

    p = _label_probs(pipe, [cleaned])[0]
    probs = {lab: float(p[i]) for i, lab in enumerate(labels)}
    preds = {lab: int(p[i] >= thresholds[lab]) for i, lab in enumerate(labels)}
    is_harmful = bool(preds.get(PRIMARY_LABEL, 0))
    raw_flagged = [lab for lab in labels if lab != PRIMARY_LABEL and preds[lab] == 1]
    flagged = raw_flagged if is_harmful else []

    primary_probability = probs.get(PRIMARY_LABEL, 0.0)
    primary_threshold = thresholds.get(PRIMARY_LABEL, 0.5)
    target_positive = len(raw_flagged) > 0
    borderline = bool(
        BORDERLINE_REVIEW_ENABLED and
        (not is_harmful) and
        target_positive and
        primary_probability >= max(0.0, primary_threshold - BORDERLINE_MARGIN)
    )

    words = _influential_words(pipe, cleaned, labels, flagged)

    return {
        "probs": probs,
        "preds": preds,
        "flagged": flagged,
        "raw_flagged": raw_flagged,
        "is_harmful": is_harmful,
        "primary_probability": primary_probability,
        "primary_threshold": primary_threshold,
        "thresholds": thresholds,
        "threshold": primary_threshold,
        "borderline_review": borderline,
        "decision": "HARMFUL" if is_harmful else ("NEEDS_REVIEW" if borderline else "CLEAN"),
        "words": words,
        "cleaned": cleaned,
        "manual_threshold_override": override_used,
    }


def _get_word_vectorizer(pipeline):
    """
    Pipelines now use a FeatureUnion of a WORD-level and a CHARACTER-level
    TF-IDF vectorizer (see src/train_utils.py). For word-highlighting we only
    want the word-level one - highlighting a 3-5 character substring like
    "tup" wouldn't mean anything to someone reading the app.

    Returns (word_vectorizer, start_index, end_index) where start/end mark
    where the word vectorizer's features sit within the full concatenated
    feature vector the classifier was actually trained on - needed to slice
    out the right coefficients/importances below.
    Falls back to a plain single "tfidf" step for older model files.
    """
    if "features" in pipeline.named_steps:
        union = pipeline.named_steps["features"]
        offset = 0
        for name, transformer in union.transformer_list:
            n_feat = len(transformer.get_feature_names_out())
            if name == "word":
                return transformer, offset, offset + n_feat
            offset += n_feat
        return None, None, None
    if "tfidf" in pipeline.named_steps:
        tfidf = pipeline.named_steps["tfidf"]
        return tfidf, 0, len(tfidf.get_feature_names_out())
    return None, None, None


def _influential_words(pipeline, cleaned, labels, flagged, top_k=8):
    """Find the unigram tokens in the comment that pushed it toward its labels."""
    try:
        clf = pipeline.named_steps["clf"]
    except (AttributeError, KeyError):
        return []

    word_vec, start, end = _get_word_vectorizer(pipeline)
    if word_vec is None:
        return []

    row = word_vec.transform([cleaned])
    if row.nnz == 0:
        return []
    feat_names = word_vec.get_feature_names_out()
    nz = row.indices                      # feature indices present in this comment

    # Which label estimators to look at (the ones that fired; else all)
    target_labels = flagged if flagged else labels
    idxs = [labels.index(l) for l in target_labels]

    contrib = np.zeros(len(feat_names))
    for i in idxs:
        est = clf.estimators_[i]
        if hasattr(est, "coef_"):                 # Logistic Regression
            w_full = np.asarray(est.coef_).ravel()
        elif hasattr(est, "calibrated_classifiers_"):  # calibrated LinearSVC
            coef_rows = [
                np.asarray(cal.estimator.coef_).ravel()
                for cal in est.calibrated_classifiers_
                if hasattr(cal.estimator, "coef_")
            ]
            if not coef_rows:
                continue
            w_full = np.mean(coef_rows, axis=0)
        elif hasattr(est, "feature_importances_"):
            # Global forest importances are not local explanations and must
            # not be presented as words that drove this individual result.
            return []
        else:
            continue
        w = w_full[start:end]             # slice out just the word-feature weights
        for j in nz:
            contrib[j] += row[0, j] * w[j]

    # Rank present unigram features by positive contribution
    ranked = sorted(nz, key=lambda j: contrib[j], reverse=True)
    words = []
    for j in ranked:
        name = feat_names[j]
        if " " in name:            # skip bigrams for highlighting
            continue
        if contrib[j] <= 0:
            break
        words.append(name)
        if len(words) >= top_k:
            break
    return words


def highlight_html(original_text, influential_words):
    """Return the comment as HTML with influential words coloured red."""
    infl = set(influential_words)
    out = []
    for tok in original_text.split():
        c = clean_text(tok)
        if c and c in infl:
            out.append(f"<span style='color:#c0392b;font-weight:700'>{tok}</span>")
        else:
            out.append(tok)
    return " ".join(out)


def explain(result):
    """Build a short plain-English explanation of the prediction."""
    if result.get("borderline_review"):
        return ("Needs human review. The primary harmful-content probability is "
                f"{result['primary_probability']:.2f}, close to its {result['primary_threshold']:.2f} "
                "OOF-selected threshold, while at least one target-community label is positive.")
    if not result["is_harmful"]:
        return ("No harmful content detected. The primary harmful-content probability "
                f"({result['primary_probability']:.2f}) did not cross its "
                f"{result['primary_threshold']:.2f} OOF-selected threshold.")
    cats = [pretty(l) for l in result["flagged"]]
    words = result["words"]
    msg = "Flagged as harmful"
    if cats:
        msg += " with " + ", ".join(cats) + "."
    else:
        msg += "."
    if words:
        msg += " The strongest contributing word(s) were: " + ", ".join(words[:5]) + "."
    else:
        msg += (" No single word was presented as the cause of the decision; the "
                "result reflects the model's learned pattern over the text.")
    return msg

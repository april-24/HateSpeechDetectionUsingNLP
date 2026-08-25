"""
predictor.py
------------
The prediction engine used by the Streamlit app. It loads a saved model bundle
and, for a given comment, returns:
    - per-label probability / confidence
    - the binary prediction using the selected decision threshold
    - the words that most influenced the decision (for highlighting)
    - a short plain-English explanation

Works for all three models:
    Logistic Regression / Random Forest -> predict_proba
    Linear SVM (no predict_proba)        -> decision_function squashed to 0..1
Word influence:
    linear models -> signed coefficients (coef_)
    random forest -> feature importances
"""

import os
import numpy as np
import joblib

from .preprocessing import clean_text
from .config import LABELS, pretty, MODEL_FILES


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
    if "threshold" not in bundle:
        bundle["threshold"] = 0.50
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
    """Return an (n_texts, n_labels) array of probabilities in 0..1."""
    try:
        return np.asarray(pipeline.predict_proba(texts))
    except (AttributeError, RuntimeError):
        # Linear SVM: squash the signed distance with a logistic function.
        d = np.asarray(pipeline.decision_function(texts))
        if d.ndim == 1:
            d = d.reshape(1, -1)
        return 1.0 / (1.0 + np.exp(-d))


def predict(bundle, text, threshold=0.5):
    """
    Analyze one comment.
    Returns a dict:
        probs   : {label: probability}
        preds   : {label: 0/1}
        flagged : list of labels predicted positive
        is_flagged: bool (any label positive)
        words   : list of influential words (for highlighting)
    """
    labels = bundle["labels"]
    pipe = bundle["pipeline"]
    cleaned = clean_text(text)

    p = _label_probs(pipe, [cleaned])[0]
    probs = {lab: float(p[i]) for i, lab in enumerate(labels)}
    preds = {lab: int(p[i] >= threshold) for i, lab in enumerate(labels)}
    flagged = [lab for lab in labels if preds[lab] == 1]

    words = _influential_words(pipe, cleaned, labels, flagged)

    return {
        "probs": probs,
        "preds": preds,
        "flagged": flagged,
        "is_flagged": len(flagged) > 0,
        "words": words,
        "cleaned": cleaned,
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
        if hasattr(est, "coef_"):                 # LR / LinearSVC
            w_full = np.asarray(est.coef_).ravel()
        elif hasattr(est, "feature_importances_"):  # Random Forest
            w_full = est.feature_importances_
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
    if not result["is_flagged"]:
        return ("No hate/offensive content detected. None of the category scores crossed "
                "the 0.5 threshold, so the comment reads as non-abusive.")
    cats = [pretty(l) for l in result["flagged"]]
    words = result["words"]
    msg = "Flagged as " + ", ".join(cats) + "."
    if words:
        msg += " The prediction was driven mainly by the word(s): " + \
               ", ".join(words[:5]) + "."
    else:
        msg += (" No single word dominated — the decision came from the overall "
                "wording rather than one term.")
    return msg

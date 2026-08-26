"""Prediction engine.

The six model outputs have independent thresholds. The ``abusive`` threshold
alone controls the final YES/NO harmful-content verdict. Target labels are
contextual and cannot turn a NO into YES.
"""
import os
import numpy as np
import joblib

from .preprocessing import clean_text
from .config import LABELS, PRIMARY_LABEL, pretty, MODEL_FILES, DEFAULT_THRESHOLDS


def available_models():
    return {name: path for name, path in MODEL_FILES.items()
            if os.path.exists(path)}


def load_model(path):
    bundle = joblib.load(path)
    try:
        clf = bundle["pipeline"].named_steps.get("clf")
        estimators = getattr(clf, "estimators_", None) or []
        for est in estimators:
            if hasattr(est, "n_jobs"):
                est.n_jobs = 1
    except Exception:
        pass
    # Backward compatibility: convert an old scalar threshold bundle to a
    # six-label dictionary. New final bundles always contain "thresholds".
    if "thresholds" not in bundle:
        old = float(bundle.get("threshold", 0.5))
        bundle["thresholds"] = {lab: old for lab in bundle.get("labels", LABELS)}
    for lab in bundle.get("labels", LABELS):
        bundle["thresholds"].setdefault(lab, 0.5)
    bundle["primary_label"] = PRIMARY_LABEL
    return bundle


def _label_probs(pipeline, texts):
    if not hasattr(pipeline, "predict_proba"):
        raise TypeError(
            "Saved model has no predicted probabilities; retrain it with "
            "formal calibration before using probability thresholds."
        )
    values = np.asarray(pipeline.predict_proba(texts))
    if values.ndim == 3:
        values = values[:, :, 1]
    return values


def thresholds_for(bundle):
    return {lab: float(bundle["thresholds"].get(lab, 0.5))
            for lab in bundle["labels"]}


def predict(bundle, text, threshold=None):
    """Return separate primary verdict, threshold and target predictions.

    ``threshold`` is accepted only for backward compatibility and is ignored
    intentionally. The evaluated rule must not be user-editable.
    """
    labels = bundle["labels"]
    pipe = bundle["pipeline"]
    cleaned = clean_text(text)
    p = _label_probs(pipe, [cleaned])[0]
    thresholds = thresholds_for(bundle)
    probs = {lab: float(p[i]) for i, lab in enumerate(labels)}
    preds = {lab: int(p[i] >= thresholds[lab]) for i, lab in enumerate(labels)}

    harmful_probability = probs[PRIMARY_LABEL]
    harmful_threshold = thresholds[PRIMARY_LABEL]
    is_harmful = bool(harmful_probability >= harmful_threshold)

    target_predictions = {
        lab: preds[lab] for lab in labels if lab != PRIMARY_LABEL
    }
    target_flagged = [
        lab for lab, value in target_predictions.items() if value
    ]
    # Target categories are displayed only as context for an already harmful
    # comment. They can never independently create the harmful verdict.
    flagged = [lab for lab in target_flagged] if is_harmful else []

    words = _influential_words(
        pipe, cleaned, labels, [PRIMARY_LABEL] + flagged)

    return {
        "probs": probs,
        "preds": preds,
        "target_predictions": target_predictions,
        "target_flagged": target_flagged,
        "flagged": flagged,
        "is_harmful": is_harmful,
        "harmful_probability": harmful_probability,
        "harmful_threshold": harmful_threshold,
        "thresholds": thresholds,
        "raw_flagged": [lab for lab in labels if preds[lab]],
        "words": words,
        "cleaned": cleaned,
        "threshold": harmful_threshold,  # compatibility
    }


def _get_word_vectorizer(pipeline):
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


def _influential_words(pipeline, cleaned, labels, target_labels, top_k=8):
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
    nz = row.indices
    idxs = [labels.index(l) for l in target_labels if l in labels]
    contrib = np.zeros(len(feat_names))
    for i in idxs:
        est = clf.estimators_[i]
        if hasattr(est, "coef_"):
            w_full = np.asarray(est.coef_).ravel()
        elif hasattr(est, "calibrated_classifiers_"):
            coef_rows = [
                np.asarray(cal.estimator.coef_).ravel()
                for cal in est.calibrated_classifiers_
                if hasattr(cal.estimator, "coef_")
            ]
            if not coef_rows:
                continue
            w_full = np.mean(coef_rows, axis=0)
        elif hasattr(est, "feature_importances_"):
            return []
        else:
            continue
        w = w_full[start:end]
        for j in nz:
            contrib[j] += row[0, j] * w[j]
    ranked = sorted(nz, key=lambda j: contrib[j], reverse=True)
    words = []
    for j in ranked:
        name = feat_names[j]
        if " " in name or contrib[j] <= 0:
            continue
        words.append(name)
        if len(words) >= top_k:
            break
    return words


def highlight_html(original_text, influential_words):
    infl = set(influential_words)
    out = []
    for tok in original_text.split():
        c = clean_text(tok)
        if c and c in infl:
            out.append(
                f"<span style='color:#c0392b;font-weight:700'>{tok}</span>")
        else:
            out.append(tok)
    return " ".join(out)


def explain(result):
    if not result["is_harmful"]:
        return (
            "NO harmful content. The abusive probability "
            f"({result['harmful_probability']:.1%}) is below the fixed "
            f"abusive threshold ({result['harmful_threshold']:.2f}). "
            "Target-group predictions are contextual only."
        )
    cats = [pretty(l) for l in result["flagged"]]
    msg = (
        f"YES harmful content. The abusive probability "
        f"({result['harmful_probability']:.1%}) meets/exceeds the fixed "
        f"abusive threshold ({result['harmful_threshold']:.2f})."
    )
    if cats:
        msg += " Target-group context: " + ", ".join(cats) + "."
    if result["words"]:
        msg += " Strong contributing terms: " + ", ".join(
            result["words"][:5]) + "."
    else:
        msg += (
            " The model did not identify a single word as a sufficient local "
            "explanation."
        )
    return msg

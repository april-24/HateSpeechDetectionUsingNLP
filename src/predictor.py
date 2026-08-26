"""Prediction engine for the deployed application.

The application uses a two-stage decision:
1. Stage 1: ``abusive`` / harmful-content detection using its dedicated OOF-selected threshold.
2. Stage 2: target-community labels are identified using their own OOF-selected thresholds,
   but are only surfaced when Stage 1 is positive.

Linear SVM probabilities come from CalibratedClassifierCV, not a manual sigmoid transform.
"""

import os
import numpy as np
import joblib

from .preprocessing import clean_text
from .config import LABELS, PRIMARY_LABEL, pretty, MODEL_FILES, DEFAULT_THRESHOLDS


def available_models():
    return {name: path for name, path in MODEL_FILES.items() if os.path.exists(path)}


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
    return bundle


def _label_probs(pipeline, texts):
    if not hasattr(pipeline, "predict_proba"):
        raise TypeError("Saved model has no predicted probabilities; retrain it with calibration.")
    return np.asarray(pipeline.predict_proba(texts))


def get_label_thresholds(bundle, primary_override=None):
    thresholds = dict(bundle.get("label_thresholds", {}))
    labels = bundle.get("labels", LABELS)
    for label in labels:
        thresholds.setdefault(label, DEFAULT_THRESHOLDS.get(
            label, bundle.get("primary_threshold", bundle.get("threshold", 0.5))))
    if primary_override is not None:
        thresholds[PRIMARY_LABEL] = float(primary_override)
    return thresholds


def apply_two_stage(probabilities, labels, thresholds):
    probabilities = np.asarray(probabilities)
    raw = np.zeros_like(probabilities, dtype=int)
    for idx, label in enumerate(labels):
        raw[:, idx] = (probabilities[:, idx] >= float(thresholds[label])).astype(int)
    primary_idx = labels.index(PRIMARY_LABEL)
    final = raw.copy()
    target_indices = [i for i, label in enumerate(labels) if label != PRIMARY_LABEL]
    if target_indices:
        final[:, target_indices] = final[:, target_indices] * raw[:, [primary_idx]]
    return final, raw


def predict(bundle, text, threshold=None):
    labels = bundle["labels"]
    pipe = bundle["pipeline"]
    cleaned = clean_text(text)
    thresholds = get_label_thresholds(bundle, primary_override=threshold)
    p = _label_probs(pipe, [cleaned])[0]
    final_pred, raw_pred = apply_two_stage(p.reshape(1, -1), labels, thresholds)
    final_pred = final_pred[0]
    raw_pred = raw_pred[0]
    probs = {lab: float(p[i]) for i, lab in enumerate(labels)}
    raw_flagged = [lab for i, lab in enumerate(labels) if raw_pred[i] == 1]
    is_harmful = bool(raw_pred[labels.index(PRIMARY_LABEL)])
    flagged = [lab for lab in labels if final_pred[labels.index(lab)] == 1]
    words = _influential_words(pipe, cleaned, labels, flagged)
    return {
        "probs": probs,
        "preds": {lab: int(final_pred[i]) for i, lab in enumerate(labels)},
        "raw_preds": {lab: int(raw_pred[i]) for i, lab in enumerate(labels)},
        "flagged": flagged,
        "raw_flagged": raw_flagged,
        "is_harmful": is_harmful,
        "words": words,
        "cleaned": cleaned,
        "threshold": thresholds[PRIMARY_LABEL],
        "label_thresholds": thresholds,
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
    if "tfidf" in pipeline.named_steps:
        tfidf = pipeline.named_steps["tfidf"]
        return tfidf, 0, len(tfidf.get_feature_names_out())
    return None, None, None


def _influential_words(pipeline, cleaned, labels, flagged, top_k=8):
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
    target_labels = flagged if flagged else labels
    idxs = [labels.index(l) for l in target_labels]
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
        if " " in name:
            continue
        if contrib[j] <= 0:
            break
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
            out.append(f"<span style='color:#c0392b;font-weight:700'>{tok}</span>")
        else:
            out.append(tok)
    return " ".join(out)


def explain(result):
    if not result["is_harmful"]:
        return ("No harmful content detected. The primary harmful-content "
                f"probability did not cross the {result['threshold']:.2f} threshold.")
    cats = [pretty(l) for l in result["flagged"] if l != PRIMARY_LABEL]
    words = result["words"]
    msg = "Flagged as harmful content."
    if cats:
        msg += " Possible target group(s): " + ", ".join(cats) + "."
    if words:
        msg += " The strongest contributing word(s) were: " + ", ".join(words[:5]) + "."
    else:
        msg += (" No single word dominated — the decision came from the overall "
                "wording rather than one term.")
    return msg

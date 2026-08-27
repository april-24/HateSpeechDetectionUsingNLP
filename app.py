"""
HarmShield - Multi-Model Harmful Content Detection System
=========================================================
Streamlit application. Run from the project root:

    streamlit run app.py

Pages (top navigation bar): Home | Dataset Statistics | Data Preprocessing |
       Content Detection | Model Evaluation
"""

import os
import re
import time
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import (LABELS, PRIMARY_LABEL, pretty, SCORES_CSV, DATA_DIR, RESULTS_DIR,
                        DEFAULT_THRESHOLDS, get_default_thresholds)
from src.predictor import (available_models, load_model, predict,
                           explain, highlight_html, _label_probs)
from src.preprocessing import clean_text, clean_text_steps
from src import social

st.set_page_config(page_title="HarmShield", page_icon="🛡️", layout="wide")

PAGES = ["Home", "Dataset Statistics", "Data Preprocessing",
         "Content Detection", "Model Evaluation"]

MODEL_INFO = {
    "Logistic Regression": {
        "feature_method": "TF-IDF word n-grams (1-3), up to 8,000 features",
        "algorithm_type": "Linear classifier (One-vs-Rest, one per label)",
        "note": "Fast fitted probabilities and directly interpretable coefficients.",
    },
    "Linear SVM": {
        "feature_method": "TF-IDF word n-grams (1-3), up to 8,000 features",
        "algorithm_type": "Maximum-margin linear classifier (One-vs-Rest)",
        "note": "Strong on sparse text; probabilities use cross-validated sigmoid calibration.",
    },
    "Random Forest": {
        "feature_method": "TF-IDF (unigrams, 8,000 features)",
        "algorithm_type": "Ensemble of decision trees (One-vs-Rest)",
        "note": "Captures non-linear word combinations; probability scores run "
               "more conservative than the other models' (a known property of "
               "tree ensembles on sparse text) - its own lower default "
               "threshold compensates for this.",
    },
}

EDA_IMAGES = [
    ("eda_original_class_distribution.png", "Original class distribution"),
    ("eda_label_counts.png", "Label counts"),
    ("eda_text_length.png", "Text-length distribution"),
    ("eda_label_correlation.png", "Label correlation"),
    ("eda_positive_labels_per_comment.png", "Positive outputs per comment"),
    ("eda_target_count_distribution.png", "Target-community count distribution"),
    ("eda_targets_by_abusive_status.png", "Target labels by abusive status"),
    ("eda_length_by_original_class.png", "Length by original class"),
    ("eda_split_distribution.png", "Development vs final-test label distribution"),
]

RESULT_IMAGES = [
    ("primary_metrics_comparison.png", "Primary precision / recall / F1"),
    ("primary_error_rates.png", "Primary FPR / FNR"),
    ("per_label_f1_heatmap.png", "Per-label F1 heatmap"),
    ("per_label_precision_heatmap.png", "Per-label precision heatmap"),
    ("per_label_recall_heatmap.png", "Per-label recall heatmap"),
    ("threshold_comparison.png", "OOF-selected thresholds"),
    ("oof_vs_final_test_f1.png", "OOF versus final-test F1"),
    ("model_timing_comparison.png", "Model timing"),
    ("primary_confusion_matrices.png", "Primary confusion matrices"),
    ("confusion_matrices_multilabel_final_test.png", "Multilabel confusion matrices"),
]


# Minimal, website-style top navigation: plain text links, not big colored
# buttons. Colors deliberately use currentColor/theme variables rather than
# hardcoded hex values, so the nav stays readable in both Streamlit's light
# and dark themes (hardcoded dark-grey text was invisible on the dark theme).
NAVBAR_CSS = """
<style>
div[data-testid="stHorizontalBlock"] div.stButton > button {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    color: var(--text-color, inherit) !important;
    opacity: 0.75;
    font-size: 15px !important;
    padding: 6px 10px !important;
    width: auto !important;
}
div[data-testid="stHorizontalBlock"] div.stButton > button:hover {
    opacity: 1 !important;
    text-decoration: underline !important;
}
div[data-testid="stHorizontalBlock"] div.stButton > button p {
    font-size: 15px !important;
    color: inherit !important;
}
</style>
"""


# ============================================================== helpers
@st.cache_resource(show_spinner=False)
def get_model(path):
    return load_model(path)


def selected_threshold(model_name):
    path = MODELS.get(model_name)
    if path:
        b = get_model(path)
        return float(b.get("thresholds", {}).get(
            PRIMARY_LABEL, get_default_thresholds(model_name)[PRIMARY_LABEL]))
    return float(get_default_thresholds(model_name)[PRIMARY_LABEL])


def selected_thresholds(model_name):
    path = MODELS.get(model_name)
    defaults = get_default_thresholds(model_name)
    if path:
        saved = get_model(path).get("thresholds", {})
        return {label: float(saved.get(label, defaults.get(label, 0.5)))
                for label in LABELS}
    return defaults


@st.cache_data(show_spinner=False)
def get_dataset():
    from src.data_loader import load_dataset
    df, text_col, labels = load_dataset(DATA_DIR, verbose=False)
    return df, text_col, labels


def show_saved_image(filename, caption):
    """Display a generated result/EDA image without failing if it is absent."""
    path = RESULTS_DIR / filename
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.warning(f"Missing chart `{filename}`. Run the project result-generation command first.")


def quality_summary(raw):
    comments = raw["comment"].fillna("").astype(str)
    cleaned = comments.map(clean_text)
    return pd.DataFrame([{
        "Raw records": len(raw),
        "Missing comments": int(raw["comment"].isna().sum()),
        "Duplicate comments": int(comments.duplicated().sum()),
        "Empty after cleaning": int(cleaned.str.len().eq(0).sum()),
        "Usable after cleaning": int(cleaned.str.len().gt(0).sum()),
    }])


def analyze_many(bundle, texts, threshold=None):
    cleaned = [clean_text(t) for t in texts]
    P = _label_probs(bundle["pipeline"], cleaned)
    labels = bundle["labels"]
    thresholds = bundle["thresholds"]
    rows = []
    for i, t in enumerate(texts):
        probs = {l: float(P[i][j]) for j, l in enumerate(labels)}
        preds = {l: int(probs[l] >= float(thresholds[l])) for l in labels}
        is_harmful = bool(preds[PRIMARY_LABEL])
        target_flags = [l for l in labels if l != PRIMARY_LABEL and preds[l]]
        flagged = target_flags if is_harmful else []
        rows.append({
            "Comment": t,
            "Harmful content": "YES" if is_harmful else "NO",
            "Categories": ", ".join(pretty(l) for l in flagged) or "-",
            "Primary probability": round(probs[PRIMARY_LABEL], 3),
            "Primary threshold": round(float(thresholds[PRIMARY_LABEL]), 3),
            **{pretty(l): round(probs[l], 3) for l in labels},
        })
    return pd.DataFrame(rows)


def suggested_action(res):
    if not res["is_harmful"]:
        return ("✅ **No action needed.** This comment doesn't cross the "
                "detection threshold. If the conversation continues, it may "
                "be worth a quick re-check later, especially if the tone shifts.")
    top_conf = res["probs"][PRIMARY_LABEL]
    cats = [pretty(l) for l in res["flagged"] if l != "abusive"]
    lines = []
    if top_conf >= 0.80:
        lines.append("⚠️ **High predicted probability** — prioritise human review.")
    else:
        lines.append("🟡 **Moderate predicted probability** — have a human "
                     "review the post before taking action.")
    lines.append("**Suggested next steps:**")
    lines.append("- Save or screenshot the comment as evidence before it can "
                 "be edited or deleted.")
    lines.append("- Most platforms have a built-in **report** option for "
                 "harassment or hate speech — consider using it.")
    if cats:
        lines.append(f"- This appears to target **{', '.join(cats)}** — if "
                     "it's part of a repeated pattern against the same "
                     "person, consider escalating to a moderator, teacher, "
                     "or the platform's trust & safety team.")
    lines.append("- If it includes a direct threat of violence, treat it "
                 "seriously and contact local authorities.")
    lines.append("- In Malaysia, online harassment can also be reported to "
                 "the **Malaysian Communications and Multimedia Commission "
                 "(MCMC)** under the Online Safety Act 2025.")
    lines.append("\n_This is general guidance, not legal advice. For "
                 "situations involving real danger to someone, contact the "
                 "appropriate authorities directly._")
    return "\n\n".join(lines)


def batch_suggestion(df):
    n = len(df)
    n_bad = int((df["Harmful content"] == "YES").sum())
    if n == 0:
        return ""
    rate = n_bad / n
    if n_bad == 0:
        return "✅ **No harmful content detected in this batch.** No action needed."
    msg = f"⚠️ **{n_bad} of {n} comments ({rate:.0%}) were flagged.**\n\n"
    if rate >= 0.3:
        msg += ("This is a high proportion — consider reviewing the source "
               "(video, thread, or file) more closely, and if it's an "
               "ongoing conversation, consider flagging it to a moderator "
               "or platform trust & safety team before it escalates.")
    else:
        msg += ("Review the flagged rows individually before taking action "
               "— sort by the **Primary probability** column to prioritise the most "
               "serious ones first.")
    msg += "\n\nKeep evidence (screenshots/exports) of anything you plan to report."
    return msg


def result_card(res, model_name, elapsed, original_text):
    if res["is_harmful"]:
        st.error("### ⚠️ HARMFUL CONTENT DETECTED")
    else:
        st.success("### ✅ No harmful content detected")

    c1, c2, c3 = st.columns(3)
    c1.metric("Model used", model_name)
    primary_probability = res["probs"][PRIMARY_LABEL]
    c2.metric("Harmful-content probability", f"{primary_probability:.1%}")
    c3.metric("Processing time", f"{elapsed*1000:.0f} ms")

    st.write("**Target-group predictions (context only):**")
    target_rows = []
    for l, value in res["target_predictions"].items():
        target_rows.append({
            "Target label": pretty(l),
            "Prediction": "YES" if value else "NO",
            "Probability": f"{res['probs'][l]:.1%}",
            "Threshold": f"{res['thresholds'][l]:.2f}",
        })
    st.dataframe(pd.DataFrame(target_rows), hide_index=True, use_container_width=True)
    if res["flagged"]:
        st.write("**Target groups associated with the harmful result:**")
        for l in res["flagged"]:
            st.markdown(f"- **{pretty(l)}** — {res['probs'][l]:.1%} predicted probability")

    st.write("**Predicted probability by output:**")
    for l in LABELS:
        p = res["probs"].get(l, 0.0)
        st.write(f"{pretty(l)} — {p:.1%}")
        st.progress(min(max(p, 0.0), 1.0))

    if res["words"]:
        st.write("**Strongest contributing words highlighted:**")
        st.markdown(
            f"<div style='padding:10px;border:1px solid rgba(128,128,128,0.4);border-radius:6px'>"
            f"{highlight_html(original_text, res['words'])}</div>",
            unsafe_allow_html=True)

    st.info(f"**Why this result?** {explain(res)}")
    st.caption(f"Fixed harmful-content rule: abusive probability {res['harmful_probability']:.1%} vs threshold {res['harmful_threshold']:.2f}. "
               "Target-group thresholds are separate and contextual.")

    st.markdown("#### 🧭 Suggested next step")
    st.markdown(suggested_action(res))


def summary_charts(df, bundle=None):
    c1, c2 = st.columns(2)
    with c1:
        counts = df["Harmful content"].value_counts()
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%",
               colors=["#c0392b" if i == "YES" else "#27ae60" for i in counts.index])
        ax.set_title("Harmful Content vs Clean")
        st.pyplot(fig); plt.close(fig)
    with c2:
        cat_cols = [pretty(l) for l in LABELS]
        if bundle is not None:
            thresholds = bundle["thresholds"]
            flags = pd.Series({
                pretty(l): int((df[pretty(l)] >= float(thresholds[l])).sum())
                for l in LABELS
            })
        else:
            flags = (df[cat_cols] >= 0.5).sum()
        fig, ax = plt.subplots(figsize=(5, 4))
        flags.plot(kind="barh", ax=ax, color="#c0392b")
        ax.set_title("Detections per category")
        st.pyplot(fig); plt.close(fig)


def page_controls(show_model=True, key_prefix=""):
    if not show_model:
        return
    models_list = list(MODELS.keys())
    st.session_state.sel_model = st.selectbox(
        "Model", models_list,
        index=models_list.index(st.session_state.sel_model),
        key=f"{key_prefix}_model",
        help="Choose the trained model. Its saved OOF-selected thresholds are fixed.")
    primary_th = selected_threshold(st.session_state.sel_model)
    st.caption(
        f"Fixed evaluation rule: **abusive ≥ {primary_th:.2f} → YES; otherwise NO**. "
        "The threshold is selected from development-only out-of-fold predictions and cannot be changed here.")
    st.write("")


def detect_quality_issues(df, text_col):
    """Flag empty / too-short / repeated-character 'noisy' comments."""
    texts = df[text_col].astype(str)
    empty = texts.str.strip().eq("").sum()
    too_short = (texts.str.split().apply(len) <= 2).sum()
    # Uses Python's own re engine (via .apply) rather than pandas' vectorized
    # .str.contains(regex=True) - some pandas/PyArrow string backends use the
    # RE2 engine for that, which doesn't support backreferences like \1.
    _repeated_re = re.compile(r"(.)\1{3,}")
    repeated = texts.apply(lambda t: bool(_repeated_re.search(t))).sum()
    duplicates = df.duplicated(subset=[text_col]).sum()
    missing = df[text_col].isna().sum()
    return {
        "Missing (null) comments": int(missing),
        "Empty / blank comments": int(empty),
        "Extremely short (<=2 words)": int(too_short),
        "Repeated-character spam (e.g. 'aaaaaa')": int(repeated),
        "Duplicate comments": int(duplicates),
    }


def render_workflow_diagram():
    stages = ["Raw Text", "Text Cleaning", "Tokenization", "Stopword\nRemoval",
              "Lemmatization", "Feature\nExtraction\n(TF-IDF)", "Classification",
              "Prediction"]
    # Uses a semi-transparent grey overlay (rgba) instead of a hardcoded light
    # background - a solid light background with inherited (theme) text color
    # turned invisible (white-on-white) under Streamlit's dark theme. rgba
    # overlays stay readable against both light and dark backgrounds, and
    # text color is left to inherit rather than hardcoded.
    boxes = "".join(
        f"<div style='display:inline-block;padding:10px 14px;margin:4px;"
        f"border:1px solid rgba(128,128,128,0.4);border-radius:8px;"
        f"background:rgba(128,128,128,0.12);color:inherit;"
        f"font-size:13px;text-align:center;white-space:pre-line'>{s}</div>"
        + ("<span style='margin:0 4px;color:rgba(128,128,128,0.9)'>&#8594;</span>" if i < len(stages)-1 else "")
        for i, s in enumerate(stages)
    )
    st.markdown(f"<div style='line-height:2.6'>{boxes}</div>", unsafe_allow_html=True)


def render_wordcloud(text_series):
    """Render a word cloud image; falls back to a note if the wordcloud
    package isn't installed (pip install wordcloud)."""
    try:
        from wordcloud import WordCloud
    except ImportError:
        st.info("Install the `wordcloud` package to see this visualization: "
               "`pip install wordcloud` (already listed in requirements.txt). "
               "Showing the frequent-words bar chart below instead.")
        return False
    text = " ".join(text_series.astype(str))
    wc = WordCloud(width=900, height=350, background_color="white",
                   colormap="Reds", max_words=100).generate(text)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig); plt.close(fig)
    return True


# ============================================================== top navbar
st.markdown(NAVBAR_CSS, unsafe_allow_html=True)

MODELS = available_models()
if not MODELS:
    st.error("No trained models found. Train them first:\n\n"
             "```\npython -m models.member1_logistic_regression\n"
             "python -m models.member2_svm\n"
             "python -m models.member3_random_forest\n```")
    st.stop()

if "page" not in st.session_state:
    st.session_state.page = "Home"
if "sel_model" not in st.session_state:
    st.session_state.sel_model = list(MODELS.keys())[0]

logo_col, *nav_cols = st.columns([2.2] + [1] * len(PAGES))
with logo_col:
    st.markdown("**🛡️ HarmShield**")
for i, p in enumerate(PAGES):
    with nav_cols[i]:
        label = f"**{p}**" if st.session_state.page == p else p
        if st.button(label, key=f"nav_{p}"):
            st.session_state.page = p
            st.rerun()
st.markdown("<hr style='margin-top:0'>", unsafe_allow_html=True)

page = st.session_state.page


# ============================================================== Home
if page == "Home":
    st.title("🛡️ HarmShield")
    st.caption("Multi-model NLP system for hate and offensive content detection")

    st.markdown("### Project Introduction")
    st.write("""
Hate and offensive content can be produced at a scale that makes manual review
difficult. HarmShield uses Natural Language Processing (NLP) to prioritise
potentially harmful posts for human review and to identify the communities that
the language appears to target.
""")

    st.markdown("### Project Objectives")
    st.markdown("""
1. Detect whether a given comment contains hate or offensive content.
2. Identify **which group is targeted** (race, religion, gender, etc.), not just yes/no.
3. Implement and fairly **compare three different NLP models** on the same data.
4. Evaluate model performance using standard classification metrics.
5. Provide an explanation and a suggested next step for every prediction.
""")

    st.markdown("### NLP Task")
    st.write("""
HarmShield uses **multi-output text classification**. The primary output states
whether a post is harmful. Five secondary outputs identify possible target
communities. A target-community score never creates a harmful verdict by itself.
**Input:** raw text. **Output:** a harmful-content probability and supporting
target-community probabilities.
""")

    st.markdown("### Implemented NLP Models")
    for name, info in MODEL_INFO.items():
        if name not in MODELS:
            continue
        st.markdown(f"**{name}** — {info['algorithm_type']}. "
                    f"Feature extraction: {info['feature_method']}. {info['note']}")

    st.markdown("### Dataset Summary")
    try:
        df, text_col, labels = get_dataset()
        c1, c2, c3 = st.columns(3)
        c1.metric("Total records", f"{len(df):,}")
        c2.metric("Classes / categories", len(labels))
        c3.metric("Source", "HateXplain (Kaggle)")
        st.caption("See the **Dataset Statistics** page for the full breakdown.")
    except Exception as e:
        st.warning(f"Dataset summary unavailable: {e}")

    st.markdown("### Explore")
    nc = st.columns(4)
    targets = ["Dataset Statistics", "Data Preprocessing", "Content Detection", "Model Evaluation"]
    descs = ["See the data behind the models", "See how raw text becomes model input",
            "Try the detector yourself", "Compare model performance"]
    for i, (t, d) in enumerate(zip(targets, descs)):
        with nc[i]:
            st.markdown(f"**{t}**")
            st.caption(d)
            if st.button("Go →", key=f"home_go_{t}"):
                st.session_state.page = t
                st.rerun()

    st.caption("Educational project. Predictions are statistical and can be "
               "wrong — always apply human judgement before acting on a result.")


# ============================================================== Dataset Statistics
elif page == "Dataset Statistics":
    st.title("Dataset Statistics")

    try:
        from src.data_loader import find_csv
        source_file = os.path.basename(find_csv(DATA_DIR))
    except Exception:
        source_file = "final_hateXplain.csv"

    try:
        df, text_col, labels = get_dataset()
        raw = pd.read_csv(find_csv(DATA_DIR))
    except Exception as e:
        st.error(f"Could not load the dataset: {e}")
        st.stop()

    overview_tab, eda_tab = st.tabs(["Dataset Overview", "EDA"])

    with overview_tab:
        st.markdown("### Dataset Overview")
        lengths = df[text_col].astype(str).str.split().apply(len)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Source file", source_file)
        c2.metric("Records", f"{len(df):,}")
        c3.metric("Classes", len(labels))
        c4.metric("Avg. length", f"{lengths.mean():.1f} words")
        st.caption("Source: HateXplain — a peer-reviewed, publicly available hate-speech dataset (via Kaggle).")

        st.markdown("### Dataset Quality")
        st.dataframe(quality_summary(raw), use_container_width=True, hide_index=True)

        st.markdown("### Dataset Preview")
        st.caption("First few rows, comment text with its labels.")
        st.dataframe(df.head(10), use_container_width=True)

        st.markdown("### Dataset Information")
        info_df = pd.DataFrame({
            "Column": df.columns,
            "Data type": [str(df[c].dtype) for c in df.columns],
            "Non-null count": [df[c].notna().sum() for c in df.columns],
        })
        st.dataframe(info_df, use_container_width=True)
        st.caption(f"Memory usage: {df.memory_usage(deep=True).sum() / 1e6:.2f} MB")

        st.markdown("### Class Distribution (Abusive vs Clean)")
        c1, c2 = st.columns(2)
        with c1:
            counts = df["abusive"].value_counts().rename({0: "Clean", 1: "Abusive"})
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%",
                   colors=["#27ae60", "#c0392b"])
            st.pyplot(fig); plt.close(fig)
        with c2:
            fig, ax = plt.subplots(figsize=(4.5, 4))
            counts.plot(kind="bar", ax=ax, color=["#27ae60", "#c0392b"])
            ax.set_ylabel("Count")
            st.pyplot(fig); plt.close(fig)

        st.markdown("### Offensive Category Distribution")
        cat_counts = df[labels].sum().sort_values(ascending=False)
        cat_counts.index = [pretty(i) for i in cat_counts.index]
        st.bar_chart(cat_counts)

        st.markdown("### Sentence Length Distribution")
        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.hist(lengths, bins=50, color="#2980b9")
        ax.set_xlim(0, lengths.quantile(0.99))
        ax.set_xlabel("Words per comment")
        st.pyplot(fig); plt.close(fig)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Min", int(lengths.min()))
        c2.metric("Median", int(lengths.median()))
        c3.metric("Mean", f"{lengths.mean():.1f}")
        c4.metric("Max", int(lengths.max()))

        st.markdown("### Most Frequent Words")
        from collections import Counter
        sample_for_cloud = df[text_col].sample(min(4000, len(df)), random_state=42).apply(clean_text)
        words = Counter(" ".join(sample_for_cloud).split())
        top = pd.Series(dict(words.most_common(25)))
        st.bar_chart(top)

        st.markdown("### Label Distribution Summary")
        dist = pd.DataFrame({
            "Category": [pretty(l) for l in labels],
            "Count": [int(df[l].sum()) for l in labels],
            "Percentage": [f"{df[l].mean():.1%}" for l in labels],
        })
        st.dataframe(dist, use_container_width=True)

        st.markdown("### NLP Workflow Overview")
        st.caption("From raw comment to final prediction:")
        render_workflow_diagram()

    with eda_tab:
        st.markdown("### Generated EDA Charts")
        st.caption("These figures are generated by `run_eda.py` and loaded from the `results/` directory; the app does not retrain or regenerate the dataset analysis at startup.")
        for filename, caption in EDA_IMAGES:
            show_saved_image(filename, caption)
        quality_path = RESULTS_DIR / "eda_dataset_quality.csv"
        if quality_path.exists():
            st.markdown("### Dataset Quality Summary")
            st.dataframe(pd.read_csv(quality_path), use_container_width=True, hide_index=True)
        else:
            st.warning("Missing `results/eda_dataset_quality.csv`.")


# ============================================================== Data Preprocessing
elif page == "Data Preprocessing":
    st.title("Data Preprocessing")
    st.caption("How raw comments are cleaned and prepared before the models see them.")

    try:
        df, text_col, labels = get_dataset()
    except Exception as e:
        st.error(f"Could not load the dataset: {e}")
        st.stop()

    st.markdown("### Dataset Quality Assessment")
    issues = detect_quality_issues(df, text_col)
    cols = st.columns(len(issues))
    for i, (k, v) in enumerate(issues.items()):
        cols[i].metric(k, f"{v:,}")
    st.caption("These are checked (and where present, handled) before the "
              "data reaches the models.")

    st.markdown("### Missing Value Handling")
    before_n = len(df) + issues["Missing (null) comments"]
    st.write(f"Rows before dropping missing comments: **{before_n:,}** → "
            f"after: **{len(df):,}** "
            f"({issues['Missing (null) comments']} removed).")
    st.caption("Handled in `src/data_loader.py` — rows with no comment text "
              "are dropped before anything else runs.")

    st.markdown("### Duplicate Detection & Removal")
    dupes = df[df.duplicated(subset=[text_col], keep=False)].head(5)
    if len(dupes):
        st.write("Example duplicate comments found in the raw data:")
        st.dataframe(dupes[[text_col]], use_container_width=True)
    else:
        st.write("No duplicate comments found in a quick scan of this dataset.")
    st.caption("Duplicates are removed during merging (`crawler/merge_datasets.py`) "
              "and before model training.")

    st.markdown("### Text Cleaning, Tokenization & Lemmatization — Live Demo")
    st.write("Pick a sample comment, or type your own, to see every "
            "preprocessing step applied to it in order.")
    demo_source = st.radio("Comment source", ["Pick from dataset", "Type my own"],
                           horizontal=True,
                           help="See the pipeline applied to a real dataset "
                                "example, or test your own sentence.")
    if demo_source == "Pick from dataset":
        sample_row = df.sample(1, random_state=None).iloc[0]
        demo_text = sample_row[text_col]
        if st.button("🔀 Shuffle — pick another random comment"):
            st.rerun()
    else:
        demo_text = st.text_input("Type a comment to preprocess",
                                  "You are SO stupid!!! @someone check http://x.com #loser",
                                  help="See exactly how this pipeline cleans your text.")

    steps = clean_text_steps(demo_text)
    for step_name, step_value in steps.items():
        st.markdown(f"**{step_name}**")
        st.code(step_value if step_value else "(empty)", language=None)

    st.markdown("### Feature Extraction (TF-IDF: word + character n-grams)")
    st.write("The cleaned text above is converted into numbers using TF-IDF "
            "(Term Frequency – Inverse Document Frequency), which weighs "
            "words by how distinctive they are, not just how often they appear. "
            "Two kinds of features are extracted: whole **words** (shown below) "
            "and **character n-grams** (3-5 letter chunks) — the character "
            "features are what let the models catch lightly disguised or "
            "misspelled words that don't match any known word exactly.")
    try:
        from src.predictor import _get_word_vectorizer
        bundle = get_model(MODELS[st.session_state.get("sel_model", list(MODELS.keys())[0])])
        cleaned_final = steps["8. Final cleaned text (fed to the model)"]
        word_vec, _, _ = _get_word_vectorizer(bundle["pipeline"])
        if cleaned_final.strip() and word_vec is not None:
            vec = word_vec.transform([cleaned_final])
            feat_names = word_vec.get_feature_names_out()
            nz = vec.nonzero()[1]
            if len(nz):
                tfidf_df = pd.DataFrame({
                    "Term": [feat_names[i] for i in nz],
                    "TF-IDF weight": [round(vec[0, i], 4) for i in nz],
                }).sort_values("TF-IDF weight", ascending=False)
                st.dataframe(tfidf_df, use_container_width=True)
            else:
                st.caption("None of these words are in the model's word-level vocabulary.")

            # Illustrative char n-gram example (not the full ~6-10k feature
            # vocabulary - just enough to show what the model actually sees).
            union = bundle["pipeline"].named_steps.get("features")
            if union is not None:
                char_vec = dict(union.transformer_list).get("char")
                if char_vec is not None:
                    cvec = char_vec.transform([cleaned_final])
                    cfeat = char_vec.get_feature_names_out()
                    cnz = cvec.nonzero()[1]
                    if len(cnz):
                        example = pd.DataFrame({
                            "Character n-gram": [cfeat[i] for i in cnz],
                            "TF-IDF weight": [round(cvec[0, i], 4) for i in cnz],
                        }).sort_values("TF-IDF weight", ascending=False).head(10)
                        st.caption("Example character n-grams extracted (top 10 by weight):")
                        st.dataframe(example, use_container_width=True)
        else:
            st.caption("Nothing left to vectorize after cleaning.")
    except Exception as e:
        st.caption(f"TF-IDF preview unavailable: {e}")

    st.markdown("### Outlier Handling")
    repeated_count = issues["Repeated-character spam (e.g. 'aaaaaa')"]
    st.write(f"- **{issues['Extremely short (<=2 words)']:,}** comments have "
            "2 words or fewer after basic cleaning (low signal for classification).")
    st.write(f"- **{repeated_count:,}** comments contain repeated-character spam patterns.")
    st.caption("Flagged for awareness; not automatically removed from "
              "training, since even short comments can be genuinely abusive "
              "(e.g. \"kill yourself\").")

    st.markdown("### Before & After Comparison")
    demo_df = df.sample(min(5, len(df)), random_state=1)[[text_col]].copy()
    demo_df["Cleaned"] = demo_df[text_col].apply(clean_text)
    demo_df.columns = ["Original", "Cleaned"]
    st.dataframe(demo_df, use_container_width=True)

    st.markdown("### Processed Dataset Preview")
    preview = df.head(10).copy()
    preview["cleaned_" + text_col] = preview[text_col].apply(clean_text)
    st.dataframe(preview, use_container_width=True)


# ============================================================== Content Detection
elif page == "Content Detection":
    st.title("Hate and Offensive Content Detection")
    page_controls(show_model=True, key_prefix="detect")
    bundle = get_model(MODELS[st.session_state.sel_model])
    threshold = float(bundle["thresholds"][PRIMARY_LABEL])
    model_name = st.session_state.sel_model

    tab1, tab2, tab3 = st.tabs(["✍️ Enter Comment", "📁 Import CSV", "🌐 Social Media URL"])

    # ---- Tab 1: Enter Comment ----
    with tab1:
        if "text_input" not in st.session_state:
            st.session_state.text_input = ""
        c1, c2 = st.columns([1, 1])
        if c1.button("Load example"):
            st.session_state.text_input = (
                "you are a stupid idiot nobody likes you\n"
                "Thanks for sharing, this was really useful!\n"
                "go back to your own country you don't belong here")
        if c2.button("Clear"):
            st.session_state.text_input = ""

        text = st.text_area(
            "Comment(s)", key="text_input", height=140,
            placeholder="Type a comment here...",
            help="One comment, or paste several — put each on its own line "
                 "to analyse them all at once.")

        if st.button("Analyze", type="primary", key="analyze_text"):
            lines = [l.strip() for l in text.split("\n") if l.strip()]
            if not lines:
                st.warning("Please enter at least one comment.")
            elif len(lines) == 1:
                t0 = time.time()
                res = predict(bundle, lines[0])
                result_card(res, model_name, time.time() - t0, lines[0])
            else:
                st.write(f"Analyzing **{len(lines)}** comments with **{model_name}**...")
                df_res = analyze_many(bundle, lines)
                n_bad = (df_res["Harmful content"] == "YES").sum()
                c1, c2, c3 = st.columns(3)
                c1.metric("Total", len(df_res))
                c2.metric("Flagged", int(n_bad))
                c3.metric("Clean", int(len(df_res) - n_bad))
                st.dataframe(df_res[["Comment", "Harmful content", "Categories",
                                     "Primary probability"]], use_container_width=True)
                summary_charts(df_res, bundle)
                st.markdown("#### 🧭 Suggested next step")
                st.markdown(batch_suggestion(df_res))
                st.download_button("Download results (CSV)",
                                   df_res.to_csv(index=False).encode(),
                                   "harmshield_results.csv", "text/csv")

    # ---- Tab 2: Import CSV ----
    with tab2:
        up = st.file_uploader(
            "Choose a file", type=["csv", "txt"],
            help="CSV needs a text column (you'll pick which one). TXT: one "
                 "comment per line.")
        if up is not None:
            try:
                if up.name.lower().endswith(".csv"):
                    raw = pd.read_csv(up)
                    st.write("Preview:")
                    st.dataframe(raw.head(), use_container_width=True)
                    col = st.selectbox("Which column holds the comment text?",
                                       list(raw.columns),
                                       help="Pick the column containing the "
                                            "actual comment/message text.")
                    texts = raw[col].dropna().astype(str).tolist()
                else:
                    texts = [l.strip() for l in
                             up.read().decode("utf-8", errors="ignore").split("\n")
                             if l.strip()]
                st.success(f"Loaded {len(texts)} comments.")

                if st.button("Analyze file", type="primary"):
                    with st.spinner("Analyzing..."):
                        df_res = analyze_many(bundle, texts)
                    n_bad = (df_res["Harmful content"] == "YES").sum()
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Total", len(df_res))
                    c2.metric("Flagged", int(n_bad))
                    c3.metric("Flag rate", f"{n_bad/max(len(df_res),1):.0%}")
                    st.dataframe(df_res, use_container_width=True)
                    summary_charts(df_res, bundle)
                    st.markdown("#### 🧭 Suggested next step")
                    st.markdown(batch_suggestion(df_res))
                    st.download_button("Download full results (CSV)",
                                       df_res.to_csv(index=False).encode(),
                                       "batch_results.csv", "text/csv")
            except Exception as e:
                st.error(f"Could not read that file: {e}")

    # ---- Tab 3: Social Media URL ----
    with tab3:
        st.info("**Supported:** YouTube and Reddit, via official public APIs. "
               "**Not supported:** Facebook, Instagram, X/Twitter, TikTok — "
               "their Terms of Service prohibit automated collection.")
        st.caption("YouTube's API is genuinely free (no credit card) — see "
                  "the box below for a 2-minute setup, or skip it with Demo mode.")
        with st.expander("Get a free YouTube API key (~2 minutes)"):
            st.markdown("""
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create or select a project (any name is fine)
3. Search **"YouTube Data API v3"** in the API Library → click **Enable**
4. Go to **Credentials → Create Credentials → API key** → copy it
5. Paste it in the box below
""")
        api_key = st.text_input(
            "YouTube API key", type="password",
            help="Only needed for YouTube links. Leave blank for Reddit or Demo mode.")

        url = st.text_input(
            "Social media URL",
            placeholder="https://www.youtube.com/watch?v=...  or  https://www.reddit.com/r/.../comments/...",
            help="Paste a YouTube video link or a Reddit thread link.")
        n_max = st.slider("Max comments to fetch", 10, 100, 40, 10,
                          help="More comments = more thorough, but slower to fetch.")

        c1, c2 = st.columns([1, 1])
        fetch_clicked = c1.button("Fetch comments", type="primary")
        demo_clicked = c2.button("Use demo comments (no API needed)")

        if demo_clicked:
            st.session_state.fetched = social.demo_comments()
            st.session_state.is_demo = True
        if fetch_clicked:
            if not url.strip():
                st.warning("Please paste a URL first.")
            else:
                with st.spinner("Fetching..."):
                    comments, err, platform = social.fetch_comments(url, api_key, n_max)
                if err:
                    st.error(err)
                else:
                    st.session_state.fetched = comments
                    st.session_state.is_demo = False
                    st.success(f"Fetched {len(comments)} comments from {platform}.")

        comments = st.session_state.get("fetched", [])
        if comments:
            if st.session_state.get("is_demo"):
                st.warning("Showing **demo sample comments** — not real fetched data.")
            st.metric("Comments retrieved", len(comments))
            with st.expander("Preview retrieved comments"):
                for c in comments[:15]:
                    st.write("-", c)

            if st.button("Analyze comments", type="primary", key="analyze_social"):
                df_res = analyze_many(bundle, comments)
                n_bad = (df_res["Harmful content"] == "YES").sum()
                c1, c2, c3 = st.columns(3)
                c1.metric("Analyzed", len(df_res))
                c2.metric("Flagged", int(n_bad))
                c3.metric("Flag rate", f"{n_bad/len(df_res):.0%}")
                st.dataframe(df_res[["Comment", "Harmful content", "Categories",
                                     "Primary probability"]], use_container_width=True)
                summary_charts(df_res, bundle)
                st.markdown("#### 🧭 Suggested next step")
                st.markdown(batch_suggestion(df_res))
                st.download_button("Download results (CSV)",
                                   df_res.to_csv(index=False).encode(),
                                   "social_results.csv", "text/csv")


# ============================================================== Model Evaluation
elif page == "Model Evaluation":
    st.title("Model Evaluation")

    comparison_tab, charts_tab = st.tabs(["Model Comparison", "Result Charts"])

    with comparison_tab:
        st.markdown("### Model Overview")
        overview = pd.DataFrame([
            {"Model": name, "Feature Extraction": info["feature_method"],
             "Algorithm Type": info["algorithm_type"]}
            for name, info in MODEL_INFO.items() if name in MODELS
        ])
        st.dataframe(overview, use_container_width=True)

        st.markdown("### Same-Comment Prediction (all models)")
        st.caption("Each model is judged against its own evidence-based threshold rather than one shared value.")
        text = st.text_input("Comment to compare", "you are a stupid idiot nobody likes you",
                             help=f"Runs this exact comment through all {len(MODELS)} available models so you can compare their predictions.")
        if st.button("Compare models", type="primary") and text.strip():
            rows = []
            for name, path in MODELS.items():
                b = get_model(path)
                model_threshold = selected_threshold(name)
                t0 = time.time()
                r = predict(b, text)
                primary_probability = r["probs"][PRIMARY_LABEL]
                rows.append({
                    "Model": name,
                    "Threshold used": model_threshold,
                    "Prediction": "HARMFUL" if r["is_harmful"] else "Clean",
                    "Categories": ", ".join(pretty(l) for l in r["flagged"]) or "-",
                    "Harmful-content probability": round(primary_probability, 3),
                    "Time (ms)": round((time.time() - t0) * 1000, 1),
                    **{pretty(l): round(r["probs"][l], 3) for l in LABELS},
                })
            cmp = pd.DataFrame(rows)
            st.dataframe(cmp[["Model", "Threshold used", "Prediction", "Categories",
                              "Harmful-content probability", "Time (ms)"]], use_container_width=True)
            st.bar_chart(cmp.set_index("Model")[[pretty(l) for l in LABELS]].T)
            if cmp["Prediction"].nunique() > 1:
                st.warning("The models disagree on this comment.")
            else:
                st.success(f"All {len(MODELS)} models agree on this comment.")

        st.markdown("### Evaluation Metrics")
        st.caption("Accuracy is per-label accuracy averaged across all six categories. Subset accuracy is the stricter all-six-correct-at-once measure.")

        with st.expander("ℹ️ Why does each model use a different detection threshold?"):
            st.markdown("""
Each model stores six thresholds, selected from pooled out-of-fold predictions on the 80% development set. The `abusive` threshold is selected using the primary harmful-content F1 objective; each target-community threshold is selected independently using that label's F1. The final 20% test set is never used for threshold selection.

Only `abusive` controls the final YES/NO verdict. Race, Religion, Gender, Sexual Orientation and Miscellaneous are contextual target predictions and can never turn a NO into YES. The UI therefore does not provide a manual threshold slider.
""")

        if not os.path.exists(SCORES_CSV):
            st.warning("No scores yet — train the models first.")
            st.stop()

        scores = pd.read_csv(SCORES_CSV)
        display_cols = ["model", "accuracy", "subset_accuracy",
                        "precision_macro", "recall_macro", "f1_macro",
                        "f1_weighted", "train_time_sec", "predict_time_sec"]
        display_cols = [c for c in display_cols if c in scores.columns]
        st.dataframe(scores[display_cols], use_container_width=True)

        st.markdown("### Confusion Matrix")
        st.caption("Computed live on the test set for the model selected above.")
        if st.button("Compute confusion matrices & classification report"):
            from sklearn.metrics import confusion_matrix, classification_report
            from src.common import prepare_data
            bundle = get_model(MODELS[st.session_state.sel_model])
            with st.spinner("Running the model over the test set..."):
                X_train, X_test, y_train, y_test, labels = prepare_data(DATA_DIR, verbose=False)
                P = _label_probs(bundle["pipeline"], list(X_test))
                pred = np.column_stack([P[:, i] >= float(bundle["thresholds"][labels[i]]) for i in range(len(labels))]).astype(int)
            cols = st.columns(3)
            for i, l in enumerate(labels):
                cm = confusion_matrix(y_test.values[:, i], pred[:, i])
                fig, ax = plt.subplots(figsize=(2.6, 2.4))
                ax.imshow(cm, cmap="Blues")
                for a in range(2):
                    for b in range(2):
                        ax.text(b, a, cm[a, b], ha="center", va="center", fontsize=9)
                ax.set_title(pretty(l), fontsize=9)
                ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
                ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
                cols[i % 3].pyplot(fig); plt.close(fig)

            st.markdown("### Classification Report")
            report = classification_report(y_test.values, pred, target_names=labels,
                                           output_dict=True, zero_division=0)
            report_df = pd.DataFrame(report).T.round(3)
            st.dataframe(report_df, use_container_width=True)

        st.markdown("### Performance Visualization")
        metric_choice = st.selectbox("Metric to compare",
                                     ["accuracy", "f1_macro", "f1_weighted",
                                      "train_time_sec", "predict_time_sec"],
                                     help=f"Pick which metric to chart across all {len(MODELS)} models.")
        if metric_choice in scores.columns:
            st.bar_chart(scores.set_index("model")[metric_choice])

        st.markdown("### Overall Evaluation Summary")
        best_acc = scores.loc[scores["accuracy"].idxmax(), "model"]
        best_f1 = scores.loc[scores["harmful_f1"].idxmax(), "model"]
        fastest_train = (scores.dropna(subset=["train_time_sec"]).loc[
            scores.dropna(subset=["train_time_sec"])["train_time_sec"].idxmin(), "model"]
            if "train_time_sec" in scores and scores["train_time_sec"].notna().any() else "Not recorded")
        fastest_predict = (scores.dropna(subset=["predict_time_sec"]).loc[
            scores.dropna(subset=["predict_time_sec"])["predict_time_sec"].idxmin(), "model"]
            if "predict_time_sec" in scores and scores["predict_time_sec"].notna().any() else "Not recorded")

        st.markdown(f"""
- **Highest accuracy:** {best_acc}
- **Best primary harmful-content F1:** {best_f1}
- **Fastest to train:** {fastest_train}
- **Fastest to predict:** {fastest_predict}

**Strengths & weaknesses:**
- **Logistic Regression** — fast, interpretable and strong overall; a solid default choice.
- **Linear SVM** — competitive on sparse TF-IDF features; its native margins are converted to probabilities using cross-validated sigmoid calibration.
- **Random Forest** — often the highest raw accuracy and precision, at the cost of slower training and prediction.
""")

    with charts_tab:
        st.markdown("### Generated Model-Result Charts")
        st.caption("These figures are generated from the saved evaluation evidence. Opening this tab does not retrain models or recalculate metrics.")
        for filename, caption in RESULT_IMAGES:
            show_saved_image(filename, caption)


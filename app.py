"""HarmShield Streamlit prototype; loads saved results and model bundles only."""
from pathlib import Path
import time
import pandas as pd
import streamlit as st

from src.config import DATA_DIR, LABELS, PRIMARY_LABEL, RESULTS_DIR, SCORES_CSV, pretty
from src.data_loader import find_csv
from src.predictor import available_models, load_model, predict, _get_word_vectorizer
from src.preprocessing import clean_text, clean_text_steps

st.set_page_config(page_title="HarmShield", page_icon="🛡️", layout="wide")
PAGES = ["Home", "Dataset Statistics", "Data Preprocessing", "EDA & Results", "Content Detection", "Model Evaluation"]
MODEL_INFO = {
    "Logistic Regression": "Word TF-IDF 1–3 grams + character TF-IDF 3–5 grams; One-vs-Rest Logistic Regression.",
    "Linear SVM": "Word TF-IDF 1–3 grams + character TF-IDF 3–5 grams; calibrated LinearSVC in One-vs-Rest.",
    "Random Forest": "Word TF-IDF 1–3 grams; One-vs-Rest Random Forest with fixed random seed.",
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

@st.cache_resource(show_spinner=False)
def get_model(path):
    return load_model(path)

@st.cache_data(show_spinner=False)
def get_dataset():
    from src.data_loader import load_dataset
    return load_dataset(DATA_DIR, verbose=False)

MODELS = available_models()
if "page" not in st.session_state: st.session_state.page = "Home"
if MODELS and "sel_model" not in st.session_state: st.session_state.sel_model = list(MODELS)[0]


def show_image(filename, caption):
    path = RESULTS_DIR / filename
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.warning(f"Missing chart `{filename}`. Generate it with the project commands in README.md.")


def scores_table():
    p = Path(SCORES_CSV)
    if not p.exists():
        st.warning("`results/model_scores.csv` is missing. The app will not retrain models automatically.")
        return None
    order = {m:i for i,m in enumerate(["Logistic Regression", "Linear SVM", "Random Forest"])}
    s = pd.read_csv(p); s["_order"] = s["model"].map(order)
    return s.sort_values("_order").drop(columns="_order")


def model_selector(key):
    selected = st.selectbox("Model", list(MODELS), index=list(MODELS).index(st.session_state.sel_model), key=key)
    st.session_state.sel_model = selected
    bundle = get_model(MODELS[selected])
    th = float(bundle["thresholds"][PRIMARY_LABEL])
    st.caption(f"Fixed rule: **abusive ≥ {th:.2f} → YES; otherwise NO**. Target labels are contextual only.")
    return bundle


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

if not MODELS:
    st.error("No saved model bundles found under results/. Startup will not retrain models.")
    st.stop()

nav = st.columns([1.8] + [1]*len(PAGES))
with nav[0]: st.markdown("**🛡️ HarmShield**")
for i, p in enumerate(PAGES):
    with nav[i+1]:
        if st.button(p, key=f"nav_{i}", use_container_width=True):
            st.session_state.page = p; st.rerun()
page = st.session_state.page

if page == "Home":
    st.title("🛡️ HarmShield")
    st.caption("Hate and Offensive Content Detection with Target-Group Identification Using NLP")
    st.markdown("### Project scope")
    st.write("HarmShield detects harmful content in individual English posts and provides five target-community annotations as context. HateXplain is not a dedicated cyberbullying dataset, so this prototype does not claim to detect conversation-level cyberbullying behaviour.")
    st.markdown("### Decision rule")
    st.info("Only the `abusive` probability and its saved model-specific threshold determine YES/NO. A target-group output can never turn NO into YES.")
    st.markdown("### Leakage-safe workflow")
    st.write("80% development / 20% untouched final test → five-fold model-selection CV → separate five-fold OOF threshold selection → final development fit → one final-test evaluation.")
    for name, desc in MODEL_INFO.items(): st.markdown(f"**{name}** — {desc}")
    scores = scores_table()
    if scores is not None:
        st.dataframe(scores[["model","harmful_precision","harmful_recall","harmful_f1","harmful_false_positive_rate","f1_micro","threshold_abusive"]], use_container_width=True, hide_index=True)

elif page == "Dataset Statistics":
    st.title("Dataset Statistics")
    raw = pd.read_csv(find_csv(DATA_DIR)); df, text_col, labels = get_dataset()
    st.dataframe(quality_summary(raw), use_container_width=True, hide_index=True)
    st.markdown("### Original class distribution")
    st.bar_chart(raw["label"].astype(str).str.lower().value_counts().reindex(["normal","offensive","hatespeech"], fill_value=0))
    st.markdown("### Adapted label counts")
    st.dataframe(pd.DataFrame({"Label":[pretty(x) for x in labels],"Count":[int(df[x].sum()) for x in labels],"Prevalence":[df[x].mean() for x in labels]}), use_container_width=True, hide_index=True)
    st.markdown("### Dataset preview")
    st.dataframe(df.head(10), use_container_width=True, hide_index=True)

elif page == "Data Preprocessing":
    st.title("Data Preprocessing")
    df, text_col, _ = get_dataset(); raw = pd.read_csv(find_csv(DATA_DIR))
    st.dataframe(quality_summary(raw), use_container_width=True, hide_index=True)
    demo = st.text_area("Cleaning-stage demonstration", df.iloc[0][text_col], height=130)
    for name, value in clean_text_steps(demo).items():
        st.markdown(f"**{name}**"); st.code(value or "(empty)")
    st.caption("The final preprocessing path is deterministic and does not download NLTK resources at runtime.")
    st.markdown("### TF-IDF preview")
    try:
        bundle = get_model(MODELS[st.session_state.sel_model]); vec, _, _ = _get_word_vectorizer(bundle["pipeline"])
        if vec is not None and clean_text(demo):
            x = vec.transform([clean_text(demo)]); names = vec.get_feature_names_out(); idx = x.nonzero()[1]
            if len(idx): st.dataframe(pd.DataFrame({"Term":[names[i] for i in idx],"TF-IDF weight":[float(x[0,i]) for i in idx]}).sort_values("TF-IDF weight", ascending=False).head(20), use_container_width=True, hide_index=True)
    except Exception as exc:
        st.info(f"TF-IDF preview unavailable: {exc}")

elif page == "EDA & Results":
    st.title("EDA & Final Results")
    t1, t2, t3 = st.tabs(["EDA", "Result Charts", "Evidence Tables"])
    with t1:
        for f, c in EDA_IMAGES: show_image(f, c)
        q = RESULTS_DIR / "eda_dataset_quality.csv"
        if q.exists(): st.dataframe(pd.read_csv(q), use_container_width=True, hide_index=True)
        else: st.warning("Missing `eda_dataset_quality.csv`.")
    with t2:
        for f, c in RESULT_IMAGES: show_image(f, c)
    with t3:
        scores = scores_table()
        if scores is not None: st.dataframe(scores, use_container_width=True, hide_index=True)
        for slug, title in [("logistic_regression","Logistic Regression"),("linear_svm","Linear SVM"),("random_forest","Random Forest")]:
            p = RESULTS_DIR / f"per_label_{slug}.csv"
            if p.exists():
                st.markdown(f"#### {title}"); st.dataframe(pd.read_csv(p), use_container_width=True, hide_index=True)

elif page == "Content Detection":
    st.title("Hate and Offensive Content Detection")
    bundle = model_selector("detect_model")
    text = st.text_area("Comment", "you are a stupid idiot nobody likes you", height=120)
    if st.button("Analyse", type="primary") and text.strip():
        start = time.time(); result = predict(bundle, text); elapsed = (time.time()-start)*1000
        st.success("### YES — harmful content detected" if result["is_harmful"] else "### NO — no harmful content detected")
        c1,c2,c3=st.columns(3); c1.metric("Model",st.session_state.sel_model); c2.metric("Harmful probability",f"{result['harmful_probability']:.1%}"); c3.metric("Time",f"{elapsed:.1f} ms")
        st.dataframe(pd.DataFrame([{ "Target":pretty(l),"Prediction":"YES" if result["target_predictions"][l] else "NO","Probability":result["probs"][l],"Threshold":result["thresholds"][l]} for l in result["target_predictions"]]), use_container_width=True, hide_index=True)
        st.caption("Target-group predictions are contextual only and cannot create the primary harmful-content verdict.")

elif page == "Model Evaluation":
    st.title("Model Evaluation")
    scores = scores_table()
    if scores is None: st.stop()
    st.markdown("### Same-comment comparison")
    text = st.text_area("Comment", "you are a stupid idiot nobody likes you", height=100)
    if st.button("Compare all models", type="primary") and text.strip():
        rows=[]
        for name,path in MODELS.items():
            b=get_model(path); start=time.time(); r=predict(b,text)
            rows.append({"Model":name,"Prediction":"YES" if r["is_harmful"] else "NO","Harmful probability":r["harmful_probability"],"Saved abusive threshold":r["harmful_threshold"],"Time (ms)":(time.time()-start)*1000})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.markdown("### Final-test evidence")
    st.dataframe(scores, use_container_width=True, hide_index=True)
    st.markdown("### Result charts")
    for f,c in RESULT_IMAGES: show_image(f,c)

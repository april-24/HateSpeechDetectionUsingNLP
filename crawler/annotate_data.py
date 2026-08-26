"""
annotate_data.py
=================
A lightweight labeling tool for the group to manually tag the comments you
crawled. Crawled data has no labels - a human has to decide, for each
comment: is it abusive, and if so, which group does it target?

Run from the project root (a separate app from the main one):
    streamlit run crawler/annotate_data.py

Output: data/crawled/<source>_labeled.csv, already in the final binary schema
(comment, abusive, Race, Religion, Gender, Sexual_Orientation, Miscellaneous)
so it plugs straight into crawler/merge_datasets.py afterwards.

Tip: you do NOT need to label every comment you crawled. Even 50-100 labeled
comments is a legitimate, honest extension to your dataset - label a
realistic amount for your timeline rather than feeling obligated to do all of
them.

Working as a group: progress saves after every single comment to the same
output file, and the tool automatically skips anything already labeled when
reopened. So one member can label 50, close it, and a teammate can open the
same file afterwards and pick up exactly where it was left off - just make
sure everyone is working from the same copy of the file (e.g. push/pull via
GitHub between sessions, or share one machine/drive).
"""

import os
import glob
import pandas as pd
import streamlit as st

st.set_page_config(page_title="HarmShield - Annotation Tool", page_icon="🏷️", layout="centered")

RAW_DIR = "data/crawled"
LABELS = ["Race", "Religion", "Gender", "Sexual_Orientation", "Miscellaneous"]


def list_raw_files():
    files = glob.glob(os.path.join(RAW_DIR, "*_raw.csv"))
    return sorted(files)


def output_path_for(raw_path):
    base = os.path.basename(raw_path).replace("_raw.csv", "")
    return os.path.join(RAW_DIR, f"{base}_labeled.csv")


def load_progress(out_path):
    if os.path.exists(out_path):
        return pd.read_csv(out_path)
    return pd.DataFrame(columns=["comment", "abusive"] + LABELS)


st.title("🏷️ HarmShield Annotation Tool")
st.caption("Manually label crawled comments so they can be added to the training data. "
          "You don't need to label all of them - even a modest batch counts.")

raw_files = list_raw_files()
if not raw_files:
    st.warning(f"No raw crawled files found in `{RAW_DIR}/`. Run a crawler first "
              "(e.g. `python crawler/youtube_batch_crawler.py`).")
    st.stop()

raw_path = st.selectbox("Which crawled file do you want to label?", raw_files)
out_path = output_path_for(raw_path)

raw_df = pd.read_csv(raw_path)
raw_df = raw_df.dropna(subset=["comment"]).drop_duplicates(subset=["comment"]).reset_index(drop=True)
done_df = load_progress(out_path)
done_comments = set(done_df["comment"]) if len(done_df) else set()

remaining = raw_df[~raw_df["comment"].isin(done_comments)].reset_index(drop=True)

c1, c2, c3 = st.columns(3)
c1.metric("Total comments", len(raw_df))
c2.metric("Labeled so far", len(done_comments))
c3.metric("Remaining", len(remaining))

if len(remaining) == 0:
    st.success("All comments in this file are labeled! 🎉 Run "
              "`python crawler/merge_datasets.py` next to add them to your training data.")
    st.dataframe(done_df.tail(20), width="stretch")
    st.stop()

st.progress(len(done_comments) / max(len(raw_df), 1))

# --- current comment to label ---
row = remaining.iloc[0]
st.markdown("### Comment to label")
st.markdown(f"> {row['comment']}")

with st.form("label_form", clear_on_submit=True):
    is_abusive = st.radio("Does this comment contain hateful or offensive content?",
                          ["No", "Yes"], horizontal=True)
    st.write("If yes, which group(s) does it target? (tick any that apply)")
    cols = st.columns(len(LABELS))
    picks = {}
    for i, lab in enumerate(LABELS):
        display = lab.replace("_", " ")
        picks[lab] = cols[i].checkbox(display, key=f"chk_{lab}")

    skip = st.form_submit_button("Skip (not sure / not English-readable)")
    submit = st.form_submit_button("Save and next ➜", type="primary")

    if submit or skip:
        if skip:
            new_row = None  # don't record - leaves it for someone else / later
        else:
            new_row = {
                "comment": row["comment"],
                "abusive": 1 if is_abusive == "Yes" else 0,
            }
            for lab in LABELS:
                new_row[lab] = 1 if (is_abusive == "Yes" and picks[lab]) else 0

        if new_row is not None:
            done_df = pd.concat([done_df, pd.DataFrame([new_row])], ignore_index=True)
            done_df.to_csv(out_path, index=False)
        st.rerun()

st.divider()
with st.expander("Recently labeled (for review)"):
    st.dataframe(done_df.tail(10), width="stretch")

st.caption(f"Saving to: `{out_path}` — safe to close and resume anytime. "
          f"A teammate can open this same file and continue from where you left off.")

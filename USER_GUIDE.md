# 🛡️ CyberShield — User Guide

This guide walks through every page of the app: what it's for, what to type in,
and how to read the results. For installation and setup, see `README.md`
instead — this document assumes the app is already running
(`streamlit run app.py`).

---

## Getting around: the navigation bar

At the top of the app is a row of plain text links (like a normal website
navbar) — click any page name to switch. The current page is shown in bold.

## Inline guidance

Look for the small **(?)** icon next to input boxes, dropdowns, and sliders —
hover or tap it for a short explanation of what that specific input does.
This replaces long instruction boxes with guidance right where you need it.

## The Model and Detection Sensitivity controls

On the **Hate/Offensive Content Detection** page, you'll see two controls near the top:

| Control | What it does |
|---------|---------------|
| **Model** | Choose which trained model performs the detection: Logistic Regression, Linear SVM, or Random Forest. |
| **Detection sensitivity (threshold)** | How confident the model must be before it flags something as hate/offensive content. **Lower** = flags more comments (catches more, but more false alarms). **Higher** = stricter (fewer false alarms, may miss some). **Each model has its own default** — the slider auto-resets to that model's own evidence-based best value whenever you switch models, since different algorithms produce confidence scores on different scales. |

**Reading "Top confidence" / "Confidence":** this number always means "how
sure the system is about the verdict shown" — for both possible verdicts.
If the verdict is **Harmful Content**, a higher percentage means more sure it
*is* hate/offensive content. If the verdict is **Clean**, a higher percentage means
more sure it's *clean* (not the other way around) — so a "Clean" result
with 95% confidence is a clearer, more obvious non-issue than a "Clean"
result with 55% confidence, which was a closer call.

These apply across all three tabs on that page (Enter Comment, Import CSV,
Social Media URL). You don't need to touch them to use the app — the
defaults work well.

---

## 1. Home Page

**What it's for:** the landing page — introduces the project, its objectives,
the NLP task being solved, the three implemented models, and a quick summary
of the dataset.

**What's on it:**
- Project introduction (what hate/offensive content detection is and why it matters)
- Project objectives
- NLP task explanation (multi-label text classification: input a comment, get
  six yes/no predictions back)
- The three implemented models, with their feature extraction method
- Dataset summary (size, number of categories, source)
- Quick-jump buttons to the other four pages

**How to use it:** just read it, then click a page name in the nav bar (or
one of the "Go →" buttons at the bottom) to continue.

---

## 2. Dataset Statistics

**What it's for:** understand the data behind the models — useful for your
documentation's "Dataset" section, and for building trust in the results by
seeing where they come from. No input needed, just browse.

**What's on it:**
- Dataset overview (source file, record count, number of classes, average length)
- A preview of raw rows with their labels
- Dataset information (column types, non-null counts, memory usage)
- Class distribution (abusive vs. clean) as pie and bar charts
- Offensive category distribution — using our actual dataset's categories
  (Race, Religion, Gender, Sexual Orientation, Miscellaneous), not a generic
  example list
- Sentence length histogram and summary statistics (min/median/mean/max)
- A word cloud of the most common words after cleaning
- A frequent-words bar chart
- A label distribution summary table (count and percentage per category)
- An NLP workflow diagram showing the full pipeline from raw text to prediction

---

## 3. Data Preprocessing

**What it's for:** shows exactly how a raw comment gets turned into something
the models can use — step by step, with real examples, not just a description.

**How to use it:**
1. **Dataset Quality Assessment** — automatically shows counts of missing
   values, empty comments, very short comments, repeated-character spam, and
   duplicates found in the raw data.
2. **Missing Value / Duplicate Handling** — shows before/after counts and a
   few real duplicate examples if any exist.
3. **Live cleaning demo** — pick **"Pick from dataset"** to see a random real
   comment processed, or **"Type my own"** to test your own sentence. Either
   way, you'll see every stage applied in order: lowercasing → removing
   URLs/mentions/hashtags → removing punctuation/numbers → tokenization →
   stopword removal → lemmatization → the final cleaned text. Click **"🔀
   Shuffle"** to see a different random example.
4. **Feature Extraction (TF-IDF)** — shows the actual numeric weights the
   currently-selected model assigns to the words in your example.
5. **Outlier Handling** — a summary of how many comments are extremely short
   or contain spam-like repeated characters (flagged, not automatically
   removed, since even short comments can be genuinely abusive).
6. **Before & After Comparison** and **Processed Dataset Preview** — see
   several real comments next to their cleaned versions.

---

## 4. Hate/Offensive Content Detection

This page has **three tabs** for three different ways to check for
hate/offensive content. The **Model** and **Detection sensitivity** controls near the
top of the page apply to all three tabs.

### Tab: ✍️ Enter Comment

**What it's for:** test one sentence quickly, or paste several at once.

**Good to know:** the system recognizes common attempts to dodge detection —
leetspeak substitutions (`n1gg4`, `sh1t`), letters spaced apart
(`n.i.g.g.a`), and repeated-character spam (`stuuuupid`) are all normalized
before analysis. This isn't foolproof (evasion is an ongoing arms race), but
it catches the common cases plus, via character-level features, many patterns
that weren't explicitly anticipated.

**How to use it:**
1. Type or paste your comment(s) — **one per line** if you have more than one.
2. (Optional) Click **"Load example"** to auto-fill sample comments for a quick demo.
3. Click **"Analyze"**.
4. **One comment** → a full result card: verdict, confidence per category
   (progress bars), the specific words highlighted in red, a plain-English
   explanation, and a **suggested next step**.
   **Multiple comments** → a summary table, charts, and a CSV download.
5. Click **"Clear"** to start over.

### Tab: 📁 Import CSV

**What it's for:** analyze many comments at once from a file you already have.

**How to use it:**
1. Upload a **CSV** (pick which column holds the comment text) or a **TXT**
   file (one comment per line).
2. Click **"Analyze file"**.
3. Review the summary metrics, results table, and charts, then download the
   full results as CSV.

### Tab: 🌐 Social Media URL

**What it's for:** pull public comments directly from a YouTube video or
Reddit thread and analyze them all at once.

**What you must know:**
- ✅ **Supported: YouTube and Reddit**, via each platform's official public API.
- ❌ **Not supported: Facebook, Instagram, X/Twitter, TikTok.** X/Twitter's API
  became fully pay-per-use (no free tier at all) in February 2026. The
  others' Terms
  of Service prohibit automated collection. Paste those comments manually
  into the Enter Comment tab instead.
- 🔑 **YouTube needs a free API key — and it really is free** (no credit
  card, just a Google account). The page has a step-by-step box for getting
  one. **Reddit needs no key at all.**
- 🧪 **No key? Use "Demo comments"** — built-in sample comments so you can
  try (or demonstrate) the feature with zero setup. Always clearly labeled
  as demo data, never real.

**How to use it:**
1. (If using YouTube) paste your API key into the box provided.
2. Paste the video/thread URL, and set how many comments to fetch.
3. Click **"Fetch comments"** (or **"Use demo comments"**).
4. Expand **"Preview retrieved comments"** to skim them first if you like.
5. Click **"Analyze comments"** to run detection on all of them, then review
   the summary and download results if needed.

---

## 5. Model Evaluation

**What it's for:** the formal performance report — the metrics required for
the assignment (Accuracy, Precision, Recall, F1, Confusion Matrix), reported
the way most classification papers report them.

**What's on it:**
- **Model Overview** — each model with its feature extraction method and algorithm type.
- **Same-Comment Prediction** — type a comment and compare all available models' predictions on it side by side (each judged against its own threshold, shown in the table).
- **Evaluation Metrics table** — including:
  - **Accuracy** — the headline number, averaged per-label (each of the 6
    categories scored as its own yes/no question, then averaged). This is
    the number comparable to what most papers report.
  - **Subset Accuracy** — also shown, but this is a much stricter "all 6
    labels correct at once" measure and will always look lower — that's
    expected for multi-label systems, not a weakness.
  - Precision, Recall, F1 (macro and weighted), Training Time, Prediction Time.
- **Confusion Matrix & Classification Report** — click **"Compute confusion
  matrices & classification report"** to see per-category detail for
  whichever model is currently selected.
- **Performance Visualization** — pick a metric from the dropdown to chart it
  across all available models.
- **Overall Evaluation Summary** — auto-generated: which model has the
  highest accuracy, best macro-F1, fastest training, fastest prediction, plus
  a short strengths/weaknesses note for each model.

---

## Quick reference: which page do I need?

| I want to... | Go to |
|---|---|
| Understand what this project does | Home |
| Learn about the training data | Dataset Statistics |
| See how raw text becomes model input | Data Preprocessing |
| Test one sentence quickly | Hate/Offensive Content Detection → Enter Comment |
| Analyze a spreadsheet or text file of comments | Hate/Offensive Content Detection → Import CSV |
| Pull comments from a YouTube video or Reddit thread | Hate/Offensive Content Detection → Social Media URL |
| Get official accuracy/precision/recall/F1 numbers | Model Evaluation |

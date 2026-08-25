# 🛡️ CyberShield — Multi-Model Hate/Offensive Content Detection System

AI Assignment — **Title 4: Natural Language Processing** — TARUMT Session 202605.

A Streamlit web application that detects hate/offensive content in online comments,
identifies **which group is targeted**, reports a **confidence score**, highlights
the **influential words**, and **compares three machine-learning models**.

| Member | Method | Script |
|--------|--------|--------|
| Member 1 | **Logistic Regression** + TF-IDF | `models/member1_logistic_regression.py` |
| Member 2 | **Linear SVM** + TF-IDF | `models/member2_svm.py` |
| Member 3 | **Random Forest** + TF-IDF | `models/member3_random_forest.py` |

---

## Quick start

```bash
pip install -r requirements.txt      # 1. install
streamlit run app.py                 # 2. launch the web app
```

All three models are **already trained and included**, so the app works
immediately. To retrain them yourself:

```bash
python -m models.member1_logistic_regression
python -m models.member2_svm
python -m models.member3_random_forest
python compare_models.py
python -m src.threshold_sweep             # re-verify each model's best threshold
```

Or open **`Hate_Offensive_Content_Detection.ipynb`** in Jupyter/Colab and Run All.

---

## Application pages

Navigate using the **top navigation bar** (minimal text links, not a sidebar) —
click any page name to switch. Inline **help tooltips** (the small (?) icon)
sit right next to each input, so guidance is where you need it instead of in
a separate box.

| Page | What it does |
|------|--------------|
| **Home** | Project intro, objectives, NLP task, implemented models, dataset summary |
| **Dataset Statistics** | Dataset overview/info, class & category distribution, word cloud, NLP workflow diagram |
| **Data Preprocessing** | Live before/after demo of every cleaning step, quality checks, TF-IDF preview |
| **Hate/Offensive Content Detection** | Three tabs: Enter Comment, Import CSV, Social Media URL |
| **Model Evaluation** | Accuracy (per-label, paper-comparable), Precision/Recall/F1, confusion matrices, classification report, training/prediction time |

Every result includes: prediction, confidence score, model used, processing time,
category breakdown with progress bars, highlighted offensive words, a
plain-English explanation, and a **suggested next step** (e.g. save evidence,
report to the platform, escalate if it's a repeated pattern).

**A note on "Accuracy":** the headline **Accuracy** metric is the *per-label*
average (each of the 6 categories scored as its own yes/no question, then
averaged) — this is the number comparable to what most classification papers
report, typically 82–84% here. **Subset Accuracy** is also shown, but it's a
much stricter "all 6 labels correct at once" measure and will always look
lower for any multi-label system — that's expected, not a sign of a weak model.

---

## What it detects

The model outputs **six labels at once** (a comment can carry several):

| Label | Displayed as |
|-------|--------------|
| `abusive` | Abusive / Harmful Content |
| `Race` | Racial Hate |
| `Religion` | Religious Hate |
| `Gender` | Gender-based Attack |
| `Sexual_Orientation` | Sexual-Orientation Attack |
| `Miscellaneous` | Other Targeted Hate |

> **Note on categories:** these are the categories the **HateXplain dataset is
> actually labelled for**. Categories such as "threat" or "profanity" are *not*
> separate labels in this dataset, so the system does not claim to predict them —
> a model cannot reliably predict a class it never saw during training.

---

## Social media support — important

| Platform | Supported? | Method |
|----------|-----------|--------|
| **YouTube** | ✅ Yes | Official **YouTube Data API v3** (needs your own free key) |
| **Reddit** | ✅ Yes | Public `.json` endpoints |
| **X / Twitter** | ❌ No | As of Feb 2026, X removed its free API tier entirely — it's now pay-per-use with no free allowance (~$0.005/read, ~2M read/month cap). Some published research papers used X data collected before this change, or under special academic agreements no longer available to new developers. |
| Facebook / Instagram / TikTok | ❌ No | Their Terms of Service **prohibit** automated comment collection from arbitrary posts, and they block it technically. Meta's official Graph API is designed for managing pages/ads you own, not scraping public content. |

The app states this clearly rather than pretending otherwise. For unsupported
platforms, copy comments manually into the **Hate/Offensive Content Detection → Enter Comment** tab. A **demo
mode** with built-in sample comments is provided so the feature can be shown
without an API key — the app always labels demo data as such.

**To enable YouTube:** get a free key from Google Cloud Console (enable
"YouTube Data API v3"), then paste it into the box on the Hate/Offensive Content Detection page's Social Media URL tab — it's completely free, no credit card required.

---

## Results (untouched 20% final test set, model-specific cross-validation-selected threshold)

| Model | Per-label Accuracy | Micro-F1 | Macro-F1 | Subset Acc. | Hamming Loss | Train Time | Predict Time |
|-------|:------------------:|:--------:|:--------:|:-----------:|:------------:|:----------:|:------------:|
| Logistic Regression | **0.8332** | **0.7067** | 0.6807 | **0.3608** | 0.1668 | 8.26s | 0.38s |
| Linear SVM | 0.8235 | 0.6928 | 0.6578 | 0.3421 | 0.1765 | 11.45s | 0.36s |
| Random Forest | 0.8092 | 0.6931 | **0.6736** | 0.2951 | 0.1908 | **2.28s** | **0.18s** |

See `results/model_scores.csv` for the full metric set (precision/recall/F1
in micro, macro, *and* weighted averaging).

### Per-model detection thresholds

Thresholds are selected **only through 2-fold cross-validation inside the 80% training split** using out-of-fold micro-F1 and then frozen before the final test evaluation. The deployed models use:

| Model | CV threshold | 2-Fold CV OOF Micro-F1 | Final-test Micro-F1 |
|-------|:-------------------:|:-------------------:|:-------------------:|
| Logistic Regression | **0.48** | 0.6987 | **0.7067** |
| Linear SVM | 0.48 | 0.6785 | 0.6928 |
| Random Forest | 0.46 | 0.6956 | 0.6931 |

The final test set is not used for threshold selection or model tuning.

### A specific, measured limitation: single trigger words can dominate a prediction

Testing surfaced concrete cases worth citing directly. The standalone word
**"sand"** gets flagged as Race/Religion-related abuse by every model, and
**"white color is my favorite color"** gets flagged as Race-related by every
model — both clearly wrong readings of genuinely neutral text. This traces
directly to the training data, not a preprocessing bug:

- Of 321 HateXplain training comments containing the word "sand", **95% are
  labeled Race** — almost entirely instances of the slur "sand n-word".
- Of 3,278 comments containing "white", **47.7% are labeled Race**, well
  above the dataset's ~32.5% Race base rate.

Bag-of-words models like these have no way to separate "sand" (a compound
slur) from "sand" (a beach) — they only see word-level co-occurrence
statistics. This is a well-documented failure mode in hate-speech
classification research (see e.g. studies on "identity term bias" in
toxicity classifiers), and it's exactly why more diverse labeled training
data — including *benign* uses of commonly-flagged words — is worth
collecting (see "Want more data?" below). A related, harder-to-fix factor:
HateXplain's overall base rate for "abusive" is **61%**, so short or
ambiguous text with no strong signal either way tends to default toward the
majority class rather than "clean".
- **Subset accuracy looks low (~0.35-0.38)** because it demands *all six*
  labels be correct simultaneously. That is normal and expected for
  multi-label tasks — **Accuracy** (per-label average, ~83-85%) is the fairer,

  paper-comparable read.

---

## Evasion & obfuscation detection

People trying to slip an offensive word past a filter typically do one of
three things — all handled by `src/preprocessing.py` before the "strip
everything but letters" step, so the underlying word survives instead of
being destroyed:

| Trick | Example | Normalized to |
|-------|---------|---------------|
| Leetspeak substitution | `n1gg4`, `sh1t` | `nigga`, `shit` |
| Spaced-out letters | `n.i.g.g.a`, `s h i t` | `nigga`, `shit` |
| Repeated-character spam | `stuuuupid` | `stupid` |

A leading `@` (e.g. `@sshole`) is deliberately **stripped**, not substituted
to `a` — testing showed substituting it caused more false positives (via
garbled `@mention` text like `@something` → `asomething`) than it prevented
evasion, since genuine `@mentions` are far more common in real text than
leading-`@` evasion. `@` occurring *mid-word* (e.g. `a@@hole`) is still
substituted, since that pattern is unambiguously deliberate.

This is **not a complete solution** — evasion detection is fundamentally an
arms race, and new obfuscation tricks will always exist. As a second,
complementary layer, Logistic Regression and SVM also use **character n-gram
features** (3-5 letter chunks), which can catch obfuscation patterns the
explicit rules above don't know about, because a lightly disguised word still
shares most of its character substrings with the original. Random Forest uses
a smaller word-only feature set instead (see the Results section for why).
Both points are worth stating plainly in your report's limitations section —
being upfront about what a detection system *can't* fully solve is a
stronger academic position than implying it's solved.

---

## Known limitations (be honest about these in your report — it earns marks)

1. **Dataset domain bias.** HateXplain was collected from Twitter and Gab, which
   are hate-speech-heavy sources. Its "normal" class is therefore not everyday
   polite conversation, so ordinary benign sentences can score near the decision
   boundary.
2. **Random Forest's confidence runs systematically lower than the other
   models', even when equally correct.** See the "Per-model detection
   thresholds" note above for the full investigation and the practical fix
   (each model uses its own threshold, not one shared value).
3. **Minority categories are harder.** Gender and Miscellaneous have the
   lowest F1 across all three models — fewer training examples means weaker
   performance. This is a direct, fixable target if you add more labeled
   data via `crawler/annotate_data.py` — see "Want more data?" below.
4. **No sarcasm or context understanding.** TF-IDF is bag-of-words; it cannot
   read intent, irony, or conversation history.
5. **English only.**
6. **Not a moderation authority.** Predictions are statistical and can be wrong.
   Human review should always precede any consequential action.

---

## Want more data? Collect more English comments (optional)

The `crawler/` folder has tools to legally collect and label more comment
data, focused on English content specifically chosen to strengthen the two
weakest-performing categories (Gender and Miscellaneous) — useful as an
"extra effort" feature for a higher grade, and as a direct, measurable
response to a specific weakness rather than a generic add-on.

```bash
python crawler/youtube_batch_crawler.py     # collect (official YouTube API)
streamlit run crawler/annotate_data.py      # label what you collected
python crawler/merge_datasets.py            # merge into training data
python -m models.member1_logistic_regression  # retrain
```

**Read `crawler/CRAWLING_GUIDE.md` first** — it covers what's legal to crawl
(and what isn't), how to get free API access, and the full workflow.

---

## Project structure

```
hate_offensive_content_detection/
├── app.py                      Streamlit web application (4 pages)
├── Hate_Offensive_Content_Detection.ipynb  notebook version (train + evaluate)
├── data/
│   └── final_hateXplain.csv    dataset (20,109 comments)
├── src/
│   ├── config.py               labels + friendly display names
│   ├── data_loader.py          builds the 6 binary labels
│   ├── preprocessing.py        cleaning, tokenisation, lemmatisation
│   ├── common.py               load + clean + train/test split
│   ├── train_utils.py          shared train/evaluate/save routine
│   ├── evaluate.py             multi-label metrics
│   ├── predictor.py            confidence, word highlighting, explanations
│   └── social.py               YouTube / Reddit fetching
├── models/
│   ├── member1_logistic_regression.py
│   ├── member2_svm.py
│   └── member3_random_forest.py
├── crawler/
│   ├── CRAWLING_GUIDE.md          full guide: legal notes, setup, workflow
│   ├── youtube_batch_crawler.py   collect comments (official YouTube API)
│   ├── reddit_crawler.py          collect comments (official Reddit API)
│   ├── generic_crawler.py         robots.txt-respecting template for other sites
│   ├── annotate_data.py           Streamlit tool to label crawled comments
│   └── merge_datasets.py          combine labeled data into training set
├── run_eda.py                  EDA charts
├── compare_models.py           comparison table + chart
├── requirements.txt
└── results/                    trained models, scores, charts
```

---

## How it meets the NLP assignment requirements (a–g)

- **(a) NLP task identified** — multi-label text classification for hate/offensive content and target-community detection.
- **(b) Background study** — for the Part 1 documentation.
- **(c) Crawler *or* reliable dataset** — the assignment allows either. We use
  both: the public **HateXplain** dataset (a reliable, peer-reviewed source)
  as the base, *plus* actual crawler tools (`crawler/`) to collect additional
  real-world English comments via official APIs (YouTube, Reddit) which you
  can label and merge in — see `crawler/CRAWLING_GUIDE.md`.
- **(d) Preprocessing** — cleaning (URLs, mentions, punctuation, special
  characters), **tokenisation**, stop-word removal, **lemmatisation** (used in
  place of stemming, since it produces real dictionary words), and
  **feature extraction** via TF-IDF (unigrams + bigrams) — one of the
  assignment's named example methods.
- **(e) Each member a different method** — mapped directly onto the
  assignment's suggested method list:
  - Member 1: **Logistic Regression** (explicitly named in the spec)
  - Member 2: **Linear SVM** (explicitly named in the spec)
  - Member 3: **Random Forest** — an ensemble of **Decision Trees**, the third
    method family the spec names
- **(f) Compare & evaluate** — Accuracy, Precision, Recall, F1 (micro + macro),
  Hamming loss, and confusion matrices — all the metrics the spec requires,
  displayed live in the **Model Evaluation** and **Model Comparison** pages.
- **(g) Reliable dataset source** — HateXplain, a peer-reviewed academic dataset.

---

## Ideas for pushing toward an Excellent grade

Beyond what's already built, these are realistic, scoped additions if you
have time left before the deadline — roughly ordered by effort:

**Low effort, real payoff:**
- **Manual data augmentation for minority categories.** Gender and
  Miscellaneous have the weakest F1 scores because they have the fewest
  training examples — labeling even 100-200 more examples specifically for
  these categories (via `crawler/annotate_data.py`) would likely move these
  numbers more than any hyperparameter tweak.
- **Inter-annotator agreement.** If two group members independently label the
  same 50-100 crawled comments, compute agreement (e.g. Cohen's Kappa) between
  them. This is a genuine, citable data-quality metric that most student
  projects skip.
- **Error analysis section.** Pull 10-15 misclassified test-set comments
  (false positives and false negatives) and discuss *why* the model got them
  wrong — sarcasm, coded language, short context, etc. This is exactly the
  kind of critical analysis the rubric's "Results & Discussion" section rewards.

**Medium effort:**
- **A fourth model for contrast.** Everything here is TF-IDF-based; adding
  one model that uses a fundamentally different representation (e.g. word
  embeddings via Word2Vec/GloVe, or a small pretrained transformer like
  DistilBERT) gives a much stronger "compare different feature extraction
  approaches" narrative than three TF-IDF variants. Fair warning: a
  transformer model is a real jump in setup complexity and training time —
  scope it carefully against your remaining time.
- **Threshold tuning per label, not just globally.** Right now one threshold
  applies to all 6 categories. Tuning a separate optimal threshold per
  category (maximizing F1 per label on a cross-validation inside the training split) is a legitimate
  technique and would likely raise macro-F1 measurably.
- **Targeted data collection for weak categories.** Gender and Miscellaneous
  have the lowest F1 across all models because they have the fewest training
  examples — the crawler tools are already configured to target this
  specifically (see `crawler/CRAWLING_GUIDE.md`). Tracking Gender/Miscellaneous
  F1 before and after adding your own labeled data is concrete, citable
  evidence for your report.

**Higher effort, strong differentiator:**
- **User study / usability evaluation.** Have a few classmates or friends use
  the app and rate the explanations/suggested actions for clarity and
  trustworthiness. Real user feedback data is uncommon in student NLP
  projects and stands out.
- **Deployment monitoring angle.** Discuss (even without fully building it)
  how the system would need to be retrained/monitored over time as language
  and evasion tactics evolve — ties directly into the evasion-detection
  limitations already documented above, and shows systems-level thinking
  beyond just model accuracy.

---

## Troubleshooting

- **"No trained models found"** — run the three training commands above, and make
  sure you launched Streamlit from the project root (the folder containing `app.py`).
- **Models don't appear after retraining** — stop Streamlit (`Ctrl+C`) and start it
  again; it caches loaded models per process.
- **NLTK download errors** — the code falls back to a built-in stop-word list and
  still runs.
- **Reddit fetch fails** — Reddit rate-limits automated requests; wait and retry.


## Evaluation protocol (revised)
The benchmark uses a leakage-safe 80/20 train/final-test split with cross-validation inside the training set. TF-IDF vectorisers are fitted only on the training partition. Model-specific decision thresholds are selected using validation micro-F1 and then frozen before the final test evaluation. The final test set is not used for threshold selection or model tuning.

The system uses HateXplain-derived labels. The prototype is best described as multi-label hate/offensive-content and target-community detection. The six outputs are an adapted abusive indicator plus five target-community indicators.

# 📊 Collecting More English Comment Data — Full Guide

This guide covers how to legally collect more comment data and merge it into
your training set, focused on **English-language content specifically chosen
to strengthen the two weakest-performing categories** in this project's
evaluation: **Gender** and **Miscellaneous**.

**Workflow:** Crawl (unlabeled) → Annotate (your group labels it) → Merge →
Retrain.

> **Note:** an earlier version of this project targeted Malay-language
> content to make the system more Malaysia-relevant. That was dropped —
> reliably labeling Malay/Manglish text turned out to be difficult for a
> small group to do accurately, and English-only data collection was judged
> more effective for directly improving the categories that actually need it.

---

## 1. What's legal, and what isn't

| Source | Status | Why |
|--------|--------|-----|
| **YouTube** (official API) | ✅ Recommended | Official, sanctioned, free, instant key |
| **Reddit** (official API) | ⚠️ Usable, but slow to set up | As of Nov 2025, Reddit requires manual pre-approval for API access — even the free tier — which can take **2–4 weeks**. Apply early if you want this. |
| **Public forums / news comments** (generic crawler) | ⚠️ Case-by-case | Only if the site's `robots.txt` allows it AND their Terms of Service don't prohibit it. Check both yourself before crawling any specific site. |
| **Facebook, Instagram, X/Twitter, TikTok** | ❌ Do not scrape | Their Terms of Service explicitly prohibit automated data collection, and they block it technically. If you need comments from these platforms, copy them manually into the app's Hate/Offensive Content Detection → Enter Comment tab (small amounts only, for demo purposes). |

---

## 2. Why Gender and Miscellaneous specifically

Across all three models, these two categories consistently score the lowest
F1 — directly because they have the fewest training examples in HateXplain.
More labeled examples is the single most direct way to improve them, more
so than any hyperparameter tuning. The default search queries in both
crawler scripts are deliberately chosen around topics likely to naturally
surface this kind of discourse (gender debates, disability rights,
refugee/immigration debates) — neutral, topic-based queries rather than
anything phrased to bait hateful replies.

---

## 3. Recommended path: YouTube (start here)

**Setup (2 minutes, no waiting):**
1. Go to [console.cloud.google.com](https://console.cloud.google.com/)
2. Create/select a project
3. **APIs & Services → Library** → search "YouTube Data API v3" → **Enable**
4. **APIs & Services → Credentials → Create Credentials → API key** → copy it

**Run the crawler:**
1. Open `crawler/youtube_batch_crawler.py`
2. Paste your key into `YOUTUBE_API_KEY` (or set it as an environment variable)
3. Adjust `SEARCH_QUERIES` if you want — the defaults target gender and
   disability/refugee-related discourse specifically to strengthen the two
   weak categories
4. Run:
   ```bash
   python crawler/youtube_batch_crawler.py
   ```
5. Output: `data/crawled/youtube_english_raw.csv` — **unlabeled**, ready for
   the annotation step below.

You can re-run it any time — it automatically skips comments it already has.

---

## 4. Optional: Reddit

Apply for Reddit API access now if you want this route — approval can take
2–4 weeks under their current policy, so don't leave it to the last minute.

1. `pip install praw`
2. Register an app at [reddit.com/prefs/apps](https://www.reddit.com/prefs/apps)
   ("script" type), get your client ID and secret
3. Fill them into `crawler/reddit_crawler.py`, then:
   ```bash
   python crawler/reddit_crawler.py
   ```
4. Output: `data/crawled/reddit_english_raw.csv`

---

## 5. Optional: other public sites (generic crawler)

For a forum or news site not covered by an official API:

1. Open the target page in your browser, right-click a comment → **Inspect**,
   and note the CSS class/tag wrapping the comment text.
2. Edit `crawler/generic_crawler.py` → add your URL and selector to `TARGETS`.
3. Run:
   ```bash
   pip install requests beautifulsoup4
   python crawler/generic_crawler.py
   ```

This script **automatically checks `robots.txt` first** and refuses to crawl
any page that disallows it. It still won't tell you whether the site's Terms
of Service allow scraping — that's on you to check before adding a URL.

---

## 6. Annotate: turn raw comments into labeled data

Crawled comments have no labels yet — you have to decide, per comment,
whether it's abusive and what it targets. This is normal for a hate-speech
dataset and is genuine, gradeable work for your documentation.

**You do not need to label everything you crawled.** Even 50–100 labeled
comments is a legitimate, honest extension — label an amount that fits your
timeline, not everything you collected. **Prioritize labeling comments that
clearly touch Gender or Miscellaneous themes** — that's where the extra data
actually moves the needle.

```bash
streamlit run crawler/annotate_data.py
```

- Pick which raw file to label from the dropdown.
- For each comment: mark **Yes/No** for abusive, and tick any target
  categories that apply (Race, Religion, Gender, Sexual Orientation,
  Miscellaneous).
- Click **"Skip"** for anything you're unsure about — better to skip than
  guess.
- Progress saves after **every single comment**, so you can close it and
  resume anytime.
- **Split the work across your group.** Progress saves to the same output
  file after each comment, and the tool automatically skips anything already
  labeled — so one member can label 50, close it, and a teammate opens the
  same file afterwards and continues exactly where it was left off, with no
  overlap or duplicates.

Output: `data/crawled/<name>_labeled.csv`, already in the format the rest of
the project expects.

---

## 7. Merge into your training data

Once you've labeled at least one file:

```bash
python crawler/merge_datasets.py
```

This combines the original HateXplain data with everything in
`data/crawled/*_labeled.csv`, removes duplicates, shuffles, and saves
`data/combined_dataset.csv`. **No other code changes needed** — the loader
automatically prefers this combined file over the original once it exists.

## 8. Retrain

```bash
python -m models.member1_logistic_regression
python -m models.member2_svm
python -m models.member3_random_forest
python compare_models.py
python -m src.threshold_sweep    # re-verify each model's best threshold
```

Check `crawler/merge_datasets.py`'s printed label counts before/after, and
compare the Gender/Miscellaneous F1 scores in `results/model_scores.csv`
before and after retraining — that comparison is your evidence the extra
data actually helped, worth including directly in your report.

---

## Notes for your documentation

- **This is a legitimate way to demonstrate "extra effort" for an Excellent
  grade** — it's genuinely new data, a new skill (data annotation, API
  integration), and shows a data-driven response to a specific, measured
  weakness (rather than guessing at improvements).
- **Be honest about data quality in your report.** Manually labeled data from
  a small group has some subjectivity — mention this as a limitation.
  If time allows, having two members independently label the same subset and
  comparing agreement (inter-annotator agreement) is a strong bonus point.
- **Track the before/after numbers.** `merge_datasets.py` prints label counts,
  and `results/model_scores.csv` has per-model metrics — comparing
  Gender/Miscellaneous F1 before and after your data collection effort is
  concrete, citable evidence of impact for your report.

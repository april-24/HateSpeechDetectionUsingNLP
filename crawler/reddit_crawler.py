"""
reddit_crawler.py
==================
Collects comments from ENGLISH-language subreddits using Reddit's OFFICIAL
API (via the PRAW library) - not scraping, a proper authenticated API call.

Focused on topics likely to surface comments relevant to the two
weakest-performing categories in this project's evaluation (Gender and
Miscellaneous) - see SUBREDDITS below. (An earlier version of this project
targeted Malay-language subreddits; that was dropped as impractical for a
small group to reliably label, and English-only data collection was judged
more effective for directly improving the weak categories.)

** IMPORTANT - READ BEFORE USING **
As of November 2025, Reddit requires manual pre-approval for ALL API access,
including the free tier (their "Responsible Builder Policy"). Approval can
take 2-4 WEEKS. If your deadline is close, apply immediately, or use the
YouTube crawler instead (crawler/youtube_batch_crawler.py) which needs no
approval wait.

Setup (after approval):
    pip install praw
    1. Go to https://www.reddit.com/prefs/apps -> "create another app..."
    2. Choose type "script", fill in any name/description/redirect URI
       (redirect URI can be http://localhost:8080)
    3. Copy the client ID (under the app name) and client secret
    4. Fill in REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET below, or set them as
       environment variables (REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET)

Output: data/crawled/reddit_english_raw.csv (UNLABELED)

Run:
    python crawler/reddit_crawler.py
"""

import os
import csv
import time
from datetime import datetime

# ---------------------------------------------------------------- SETTINGS
REDDIT_CLIENT_ID = os.environ.get("REDDIT_CLIENT_ID", "PASTE_YOUR_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.environ.get("REDDIT_CLIENT_SECRET", "PASTE_YOUR_SECRET")
REDDIT_USER_AGENT = "HarmShield-Assignment/1.0 (educational project)"

# English-language subreddits chosen for topics likely to surface Gender and
# Miscellaneous (refugee/disability/other-targeted-hate) relevant comments -
# mainstream discussion communities, not ones built around hate themselves.
# Verify these are still active before running.
SUBREDDITS = ["TwoXChromosomes", "disability", "immigration", "worldnews"]

POSTS_PER_SUBREDDIT = 40      # how many recent/hot posts to pull comments from
MAX_COMMENTS_PER_POST = 30
OUTPUT_CSV = "data/crawled/reddit_english_raw.csv"
REQUEST_DELAY = 0.5           # politeness delay between posts (free tier: 100 QPM)


def main():
    try:
        import praw
    except ImportError:
        print("praw is not installed. Run:  pip install praw")
        return

    if REDDIT_CLIENT_ID.startswith("PASTE_") or REDDIT_CLIENT_SECRET.startswith("PASTE_"):
        print("ERROR: set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET before running "
              "(see the docstring at the top of this file).")
        return

    reddit = praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT,
    )
    reddit.read_only = True

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    write_header = not os.path.exists(OUTPUT_CSV)
    seen = set()
    if not write_header:
        with open(OUTPUT_CSV, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                seen.add(row["comment"])

    total_new = 0
    with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "comment", "subreddit", "post_title", "fetched_at"])
        if write_header:
            writer.writeheader()

        for sub_name in SUBREDDITS:
            print(f"\nSubreddit: r/{sub_name}")
            try:
                subreddit = reddit.subreddit(sub_name)
                posts = list(subreddit.hot(limit=POSTS_PER_SUBREDDIT))
            except Exception as e:
                print(f"  could not access r/{sub_name}: {e}")
                continue

            for post in posts:
                try:
                    post.comments.replace_more(limit=0)
                    comments = [c.body for c in post.comments.list()[:MAX_COMMENTS_PER_POST]
                               if hasattr(c, "body")]
                except Exception as e:
                    print(f"  skipped a post ({e})")
                    continue

                now = datetime.now().isoformat(timespec="seconds")
                new_here = 0
                for body in comments:
                    body = body.strip()
                    if not body or body in ("[deleted]", "[removed]") or body in seen:
                        continue
                    seen.add(body)
                    writer.writerow({"comment": body, "subreddit": sub_name,
                                     "post_title": post.title, "fetched_at": now})
                    new_here += 1
                total_new += new_here
                time.sleep(REQUEST_DELAY)

            print(f"  collected from {len(posts)} posts")

    print(f"\nDone. {total_new} new comments saved -> {OUTPUT_CSV}")
    print("Reminder: this data is UNLABELED. See crawler/CRAWLING_GUIDE.md for "
          "the annotation step before using it for training.")


if __name__ == "__main__":
    main()

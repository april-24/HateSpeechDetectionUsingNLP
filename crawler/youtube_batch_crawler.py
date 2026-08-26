"""
youtube_batch_crawler.py
=========================
Collects comments from MULTIPLE YouTube videos at once, using the official
YouTube Data API v3 - the same legitimate, ToS-compliant method the app itself
uses (see src/social.py), just scaled up for bulk data collection.

Focused on ENGLISH-language content, specifically topics likely to surface
comments relevant to the two weakest-performing categories in this project's
evaluation (Gender and Miscellaneous) - see the default SEARCH_QUERIES below.
(An earlier version of this project targeted Malay-language content; that was
dropped as impractical for a small group to reliably label, and English-only
data collection was judged more effective for directly improving the weak
categories.)

Two ways to choose videos:
  1. SEARCH_QUERIES - the script searches YouTube for videos matching each
     query and collects comments from the results.
  2. VIDEO_IDS - paste specific video URLs/IDs directly if you already know
     which videos have the discussion you want.

Get a free API key (takes 2 minutes, no approval wait):
  1. https://console.cloud.google.com/  -> create/select a project
  2. APIs & Services -> Library -> search "YouTube Data API v3" -> Enable
  3. APIs & Services -> Credentials -> Create Credentials -> API key
  4. Put the key in YOUTUBE_API_KEY below, or set it as an environment
     variable: export YOUTUBE_API_KEY=your_key_here   (Windows: set YOUTUBE_API_KEY=...)

Quota note: YouTube gives 10,000 free units/day. A search costs 100 units,
each page of up to 100 comments costs 1 unit. The defaults below stay well
under that for a single run.

Output: data/crawled/youtube_english_raw.csv (UNLABELED - see README in this folder)

Run:
    python crawler/youtube_batch_crawler.py
"""

import os
import sys
import csv
import time
import json
import urllib.request
import urllib.parse
from datetime import datetime

# ---------------------------------------------------------------- SETTINGS
YOUTUBE_API_KEY = os.environ.get("YOUTUBE_API_KEY", "AIzaSyCQoumnsgNBTL_z2aNwjd3jIGsadjRlQo4")

# Neutral, topic-based English-language queries chosen to naturally surface
# heated comment sections specifically around Gender and Miscellaneous
# (refugee/disability/other-targeted-hate) themes - the two weakest
# categories in this project's evaluation - without steering toward any
# specific target (avoid queries phrased to bait hate directly).
SEARCH_QUERIES = [
    "gender pay gap debate",
    "women in gaming controversy",
    "disability rights news",
    "immigration policy debate",
]

# Or paste specific video URLs/IDs here to skip searching entirely.
VIDEO_IDS = [
    # "https://www.youtube.com/watch?v=XXXXXXXXXXX",
]

VIDEOS_PER_QUERY = 3          # how many videos to pull comments from, per search query
MAX_COMMENTS_PER_VIDEO = 50  # up to 50 per API page
REGION = "US"                 # bias search results toward English-speaking region
OUTPUT_CSV = "data/crawled/youtube_english_raw.csv"
REQUEST_DELAY = 1.0           # seconds between API calls (politeness / quota pacing)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------- helpers
def api_get(endpoint, params):
    params["key"] = YOUTUBE_API_KEY
    url = f"https://www.googleapis.com/youtube/v3/{endpoint}?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "HarmShield-Assignment/1.0"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read().decode("utf-8"))


def search_videos(query, max_results=5):
    data = api_get("search", {
        "part": "snippet", "q": query, "type": "video",
        "maxResults": max_results, "regionCode": REGION,
        "order": "relevance",
    })
    out = []
    for item in data.get("items", []):
        vid = item.get("id", {}).get("videoId")
        title = item.get("snippet", {}).get("title", "")
        if vid:
            out.append((vid, title))
    return out


def fetch_comments(video_id, max_comments=100):
    comments, page_token = [], None
    while len(comments) < max_comments:
        params = {
            "part": "snippet", "videoId": video_id, "textFormat": "plainText",
            "maxResults": min(100, max_comments - len(comments)),
        }
        if page_token:
            params["pageToken"] = page_token
        try:
            data = api_get("commentThreads", params)
        except Exception as e:
            print(f"    (comments unavailable: {e})")
            break
        for item in data.get("items", []):
            try:
                snip = item["snippet"]["topLevelComment"]["snippet"]
                comments.append(snip.get("textDisplay", "").strip())
            except (KeyError, TypeError):
                continue
        page_token = data.get("nextPageToken")
        if not page_token:
            break
        time.sleep(REQUEST_DELAY)
    return [c for c in comments if c]


def extract_video_id(url_or_id):
    if len(url_or_id) == 11 and "/" not in url_or_id:
        return url_or_id
    import re
    for p in [r"v=([A-Za-z0-9_-]{11})", r"youtu\.be/([A-Za-z0-9_-]{11})"]:
        m = re.search(p, url_or_id)
        if m:
            return m.group(1)
    return None


# ---------------------------------------------------------------- main
def main():
    if not YOUTUBE_API_KEY or YOUTUBE_API_KEY == "PASTE_YOUR_KEY_HERE":
        print("ERROR: set YOUTUBE_API_KEY (edit this file, or set the "
              "environment variable) before running.")
        return

    videos = []  # list of (video_id, title, source_query)

    for vid_input in VIDEO_IDS:
        vid = extract_video_id(vid_input)
        if vid:
            videos.append((vid, "", "manual"))

    for q in SEARCH_QUERIES:
        print(f"Searching: {q!r}")
        try:
            found = search_videos(q, VIDEOS_PER_QUERY)
        except Exception as e:
            print(f"  search failed: {e}")
            continue
        for vid, title in found:
            videos.append((vid, title, q))
        time.sleep(REQUEST_DELAY)

    if not videos:
        print("No videos found/specified. Check your queries or VIDEO_IDS.")
        return

    print(f"\nFound {len(videos)} videos. Fetching comments...\n")

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
            "comment", "video_id", "video_title", "source_query", "fetched_at"])
        if write_header:
            writer.writeheader()

        for vid, title, query in videos:
            print(f"  Video {vid}  ({title[:50]!r})")
            try:
                comments = fetch_comments(vid, MAX_COMMENTS_PER_VIDEO)
            except Exception as e:
                print(f"    failed: {e}")
                continue
            now = datetime.now().isoformat(timespec="seconds")
            new_here = 0
            for c in comments:
                if c in seen:
                    continue
                seen.add(c)
                writer.writerow({"comment": c, "video_id": vid,
                                 "video_title": title, "source_query": query,
                                 "fetched_at": now})
                new_here += 1
            total_new += new_here
            print(f"    +{new_here} new comments")
            time.sleep(REQUEST_DELAY)

    print(f"\nDone. {total_new} new comments saved -> {OUTPUT_CSV}")
    print("Reminder: this data is UNLABELED. See crawler/CRAWLING_GUIDE.md for "
          "the annotation step before using it for training.")


if __name__ == "__main__":
    main()

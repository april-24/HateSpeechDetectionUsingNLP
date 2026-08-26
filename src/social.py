"""
social.py
---------
Fetches publicly available comments from a URL for analysis.

IMPORTANT - what is and isn't supported, and why:

  SUPPORTED (via the platform's OFFICIAL public API, with your own free key):
      * YouTube  - YouTube Data API v3
      * Reddit   - Reddit's public .json endpoints

  NOT SUPPORTED:
      * Facebook, Instagram, X/Twitter
        These platforms prohibit scraping in their Terms of Service, block
        automated access, and require private/paid API access for comment data.
        Building a scraper for them would breach their ToS, so this system does
        not do it. The app tells the user this clearly instead of pretending.

  DEMO MODE:
      If you have no API key (or are offline), `demo_comments()` returns a small
      built-in set of sample comments so the feature can still be demonstrated.
      The app always labels this clearly as demo data - it is never passed off
      as real fetched content.
"""

import re
import json
import urllib.request
import urllib.parse

USER_AGENT = "HarmShield-Assignment/2.0 (educational project)"


# --------------------------------------------------------------------------
# Platform detection
# --------------------------------------------------------------------------
def detect_platform(url: str) -> str:
    """Return 'youtube', 'reddit', 'blocked', or 'unknown' for a given URL."""
    u = (url or "").lower()
    if "youtube.com" in u or "youtu.be" in u:
        return "youtube"
    if "reddit.com" in u:
        return "reddit"
    if any(d in u for d in ["facebook.com", "fb.com", "instagram.com",
                            "twitter.com", "x.com", "tiktok.com"]):
        return "blocked"
    return "unknown"


BLOCKED_MESSAGE = (
    "This platform (Facebook / Instagram / X / TikTok) does not permit automated "
    "comment collection. Their Terms of Service prohibit scraping and they block "
    "automated access, so this system does not attempt it.\n\n"
    "Supported instead: **YouTube** and **Reddit**, using their official public APIs. "
    "You can also paste comments manually into the Text Detection page."
)


# --------------------------------------------------------------------------
# YouTube (official Data API v3 - needs a free API key)
# --------------------------------------------------------------------------
def extract_youtube_id(url: str):
    """Pull the 11-character video ID out of a YouTube URL."""
    patterns = [r"v=([A-Za-z0-9_-]{11})",
                r"youtu\.be/([A-Za-z0-9_-]{11})",
                r"shorts/([A-Za-z0-9_-]{11})"]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    return None


def fetch_youtube_comments(url: str, api_key: str, max_comments: int = 50):
    """
    Fetch top-level comments using the official YouTube Data API v3.
    Get a free key at: https://console.cloud.google.com/  (enable 'YouTube Data API v3')
    Returns (comments, error_message).
    """
    video_id = extract_youtube_id(url)
    if not video_id:
        return [], "Could not read a video ID from that URL."
    if not api_key:
        return [], "A YouTube Data API key is required (see the sidebar)."

    endpoint = "https://www.googleapis.com/youtube/v3/commentThreads?" + \
        urllib.parse.urlencode({
            "part": "snippet",
            "videoId": video_id,
            "maxResults": min(max_comments, 100),
            "textFormat": "plainText",
            "key": api_key,
        })
    try:
        req = urllib.request.Request(endpoint, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=20) as r:
            data = json.loads(r.read().decode("utf-8"))
    except Exception as e:
        return [], f"YouTube API request failed: {e}"

    comments = []
    for item in data.get("items", []):
        try:
            s = item["snippet"]["topLevelComment"]["snippet"]
            comments.append(s.get("textDisplay", "").strip())
        except (KeyError, TypeError):
            continue
    comments = [c for c in comments if c]
    if not comments:
        return [], "No comments found (comments may be disabled on that video)."
    return comments[:max_comments], None


# --------------------------------------------------------------------------
# Reddit (public .json endpoint - no key needed)
# --------------------------------------------------------------------------
def fetch_reddit_comments(url: str, max_comments: int = 50):
    """
    Fetch comments from a public Reddit thread via its .json endpoint.
    Returns (comments, error_message).
    """
    clean = url.split("?")[0].rstrip("/")
    if not clean.endswith(".json"):
        clean += ".json"
    try:
        req = urllib.request.Request(clean, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=20) as r:
            data = json.loads(r.read().decode("utf-8"))
    except Exception as e:
        return [], (f"Reddit request failed: {e}. Reddit rate-limits automated "
                    "requests; wait a moment and try again.")

    comments = []

    def walk(node):
        if len(comments) >= max_comments:
            return
        if isinstance(node, dict):
            if node.get("kind") == "t1":
                body = node.get("data", {}).get("body")
                if body and body not in ("[deleted]", "[removed]"):
                    comments.append(body.strip())
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(data)
    if not comments:
        return [], "No readable comments found in that thread."
    return comments[:max_comments], None


# --------------------------------------------------------------------------
# Demo mode (no network / no API key)
# --------------------------------------------------------------------------
def demo_comments():
    """Built-in sample comments so the feature can be demonstrated offline."""
    return [
        "This video was really helpful, thanks for posting!",
        "you are so stupid, nobody wants you here",
        "Great explanation, I finally understand the topic.",
        "go back to your own country, we don't want your kind",
        "I disagree with some points but it's a fair argument.",
        "shut up you idiot, your opinion is worthless",
        "Nice editing and clear audio, subscribed.",
        "people like you are a disease on this planet",
    ]


def fetch_comments(url, api_key=None, max_comments=50):
    """
    Unified entry point used by the app.
    Returns (comments, error_message, platform).
    """
    platform = detect_platform(url)
    if platform == "blocked":
        return [], BLOCKED_MESSAGE, platform
    if platform == "youtube":
        c, e = fetch_youtube_comments(url, api_key, max_comments)
        return c, e, platform
    if platform == "reddit":
        c, e = fetch_reddit_comments(url, max_comments)
        return c, e, platform
    return [], ("Unrecognised URL. Supported: YouTube video links and Reddit "
                "thread links."), platform

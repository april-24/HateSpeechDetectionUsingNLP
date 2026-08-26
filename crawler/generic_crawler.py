"""
generic_crawler.py
===================
A responsible, general-purpose crawler TEMPLATE for public forums or comment
sections not covered by an official API (e.g. Malaysian forums or news sites).

** BEFORE YOU POINT THIS AT ANY SITE, CHECK: **
  1. The site's robots.txt (this script checks it automatically and refuses
     to crawl a page that disallows it).
  2. The site's Terms of Service - even if robots.txt allows a path,
     automated data collection may still be against their terms. When in
     doubt, don't. Sites the assignment explicitly rules out for scraping:
     Facebook, Instagram, X/Twitter, TikTok (use the official YouTube/Reddit
     routes instead, or copy comments manually into the app's Text Detection
     page for small amounts).
  3. Only collect PUBLIC pages that don't need a login.
  4. Keep request rates low (this script already paces itself) and identify
     yourself honestly in the User-Agent.

You must edit TARGETS and the CSS selector for the site you choose - there is
no way to write one selector that works for every website's HTML structure.
Use your browser's "Inspect Element" on a comment to find the right selector.

Output: data/crawled/generic_raw.csv (UNLABELED)

Run:
    pip install requests beautifulsoup4
    python crawler/generic_crawler.py
"""

import os
import csv
import time
import urllib.request
from datetime import datetime
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

# ---------------------------------------------------------------- SETTINGS
# Each target is a page URL + the CSS selector that matches each comment's
# text on that page. EDIT THIS before running - there is no default target.
TARGETS = [
    # {"url": "https://example-forum.com/thread/123", "selector": "div.post-message"},
]

USER_AGENT = "HarmShield-Assignment-Bot/1.0 (educational project; contact: your_email@example.com)"
REQUEST_DELAY = 2.0     # seconds between page requests - be polite
OUTPUT_CSV = "data/crawled/generic_raw.csv"


def robots_allows(url: str, user_agent: str = "*") -> bool:
    """Check the site's robots.txt before touching the page."""
    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    rp = RobotFileParser()
    try:
        rp.set_url(robots_url)
        rp.read()
    except Exception:
        # If robots.txt can't be read, be conservative and refuse.
        print(f"  Could not read {robots_url} - skipping this URL to be safe.")
        return False
    return rp.can_fetch(user_agent, url)


def fetch_page(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=20) as r:
        return r.read().decode("utf-8", errors="ignore")


def extract_comments(html: str, selector: str):
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        raise RuntimeError("beautifulsoup4 is not installed. Run: pip install beautifulsoup4")
    soup = BeautifulSoup(html, "html.parser")
    return [el.get_text(strip=True) for el in soup.select(selector)]


def main():
    if not TARGETS:
        print("No TARGETS configured. Edit crawler/generic_crawler.py and add at "
              "least one {'url': ..., 'selector': ...} entry before running.")
        return

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    write_header = not os.path.exists(OUTPUT_CSV)
    seen = set()
    if not write_header:
        with open(OUTPUT_CSV, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                seen.add(row["comment"])

    total_new = 0
    with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["comment", "source_url", "fetched_at"])
        if write_header:
            writer.writeheader()

        for target in TARGETS:
            url, selector = target["url"], target["selector"]
            print(f"Checking robots.txt for {url} ...")
            if not robots_allows(url, USER_AGENT):
                print("  Disallowed by robots.txt - skipping this URL.")
                continue

            print("  Allowed. Fetching page...")
            try:
                html = fetch_page(url)
                comments = extract_comments(html, selector)
            except Exception as e:
                print(f"  Failed: {e}")
                continue

            now = datetime.now().isoformat(timespec="seconds")
            new_here = 0
            for c in comments:
                c = c.strip()
                if not c or c in seen:
                    continue
                seen.add(c)
                writer.writerow({"comment": c, "source_url": url, "fetched_at": now})
                new_here += 1
            total_new += new_here
            print(f"  +{new_here} new comments")
            time.sleep(REQUEST_DELAY)

    print(f"\nDone. {total_new} new comments saved -> {OUTPUT_CSV}")
    print("Reminder: this data is UNLABELED. See crawler/CRAWLING_GUIDE.md for "
          "the annotation step before using it for training.")


if __name__ == "__main__":
    main()

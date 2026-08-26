"""
preprocessing.py
----------------
Shared text-cleaning pipeline used by ALL three models so the comparison is fair.

Pipeline steps:
    1. lowercase
    2. strip URLs, HTML
    3. normalize evasion patterns (leetspeak, spaced-out letters, repeated chars)
    4. strip hashtag symbol, remaining digits/punctuation/special characters
    5. tokenize
    6. remove a built-in deterministic English stopword list while preserving negations
    7. retain normalized tokens without any network-dependent lemmatization

The preprocessing is deterministic and requires no NLTK resource download at startup.
"""

import re

# ----------------------------------------------------------------------------
# Deterministic preprocessing: no network downloads at import time.
# ----------------------------------------------------------------------------
# Streamlit Community Cloud may block or delay outbound NLTK downloads. The
# detector therefore uses a built-in English stopword list and deterministic
# rule-based normalization. NLTK is optional and is never downloaded here.
_USE_NLTK = False
_LEMMATIZER = None
_STOPWORDS = set("""
    a an the and or but if while is are was were be been being to of in on for with
    as by at from into this that these those it its i you he she we they them his her
    their our your my me him us do does did done have has had not no nor so than too
    very can will just don should now am im youre hes shes were theyre ive we've
""".split())

# Negation changes the meaning of moderation text ("I hate" vs "I do not
# hate"). Preserve these tokens even though standard stop-word lists remove
# them.
_STOPWORDS.difference_update({"no", "not", "nor", "never"})

# Pre-compiled regex patterns (compiled once for speed)
_RE_URL = re.compile(r"http\S+|www\.\S+")
_RE_HTML = re.compile(r"<.*?>")
_RE_NONALPHA = re.compile(r"[^a-z\s]")
_RE_SPACES = re.compile(r"\s+")

# Normalise common contractions BEFORE punctuation is stripped so negation is
# preserved consistently in every model (e.g. "don't" -> "do not").
_CONTRACTIONS = {
    "can't": "cannot", "won't": "will not", "wouldn't": "would not",
    "couldn't": "could not", "shouldn't": "should not",
    "isn't": "is not", "aren't": "are not", "wasn't": "was not",
    "weren't": "were not", "don't": "do not", "doesn't": "does not",
    "didn't": "did not", "hasn't": "has not", "haven't": "have not",
    "hadn't": "had not", "mustn't": "must not", "mightn't": "might not",
    "shan't": "shall not", "you'd": "you would", "we'd": "we would",
    "they'd": "they would", "i'd": "i would", "you'll": "you will",
    "we'll": "we will", "they'll": "they will", "i'll": "i will",
}
_RE_CONTRACTION = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in sorted(_CONTRACTIONS, key=len, reverse=True)) + r")\b"
)


# ----------------------------------------------------------------------------
# Evasion-pattern normalization
# ----------------------------------------------------------------------------
# People trying to slip an offensive word past a filter commonly do one of
# three things - all handled BEFORE the "strip everything but letters" step,
# so the underlying word survives instead of being destroyed:
#   1. Leetspeak substitution:  n1gg4, @sshole, sh1t   -> letter equivalents
#   2. Spaced-out letters:      n.i.g.g.a, s h i t      -> collapsed together
#   3. Repeated-character spam: nigggggga, stuuupid     -> collapsed down
#
# A DELIBERATE TRADE-OFF, worth noting in your report: leading '@' characters
# (e.g. "@johndoe") are stripped as mention markers, same as before evasion
# normalization was added - real-world testing showed substituting a leading
# '@' to 'a' (e.g. "@something" -> "asomething") produced garbled,
# out-of-vocabulary tokens that triggered false positives via the character
# n-gram features far more often than it caught genuine evasion. '@'
# occurring mid-word (e.g. "a@@hole") is still substituted to 'a', since
# that pattern is unambiguously a deliberate evasion attempt, not a mention.
#
# IMPORTANT (also worth a line in your report): this catches the common,
# well-known evasion patterns, but it is not a complete solution - evasion is
# an arms race, and new obfuscation tricks will always exist. The character
# n-gram features used in the models (see models/*.py) provide a second,
# complementary layer of robustness: they can catch NEW obfuscation patterns
# this normalization doesn't explicitly know about, because "n1gg4" and
# "nigga" still share overlapping 3-5 character substrings even before any
# substitution is applied.
_LEET_SUBS = {
    "@": "a", "4": "a",
    "3": "e",
    "1": "i", "|": "i",
    "0": "o",
    "5": "s", "$": "s",
    "7": "t", "+": "t",
    "9": "g",
}
# NOTE: '!' is deliberately NOT mapped here (e.g. to 'i') even though it's a
# common leet substitute in theory - in practice '!' is used FAR more often
# as genuine punctuation/emphasis ("amazing!!!"), and substituting it caused
# real false positives (e.g. "stupid!!!" -> "stupidiii") during testing.
_LEET_PATTERN = re.compile("|".join(re.escape(k) for k in _LEET_SUBS))

# 3+ identical consecutive characters -> collapsed to a SINGLE character.
# Standard English essentially never has a genuine triple-letter run (words
# like "book"/"sorry" only ever have 2 in a row, so they're untouched - this
# regex only ever fires on 3+, which in practice is always spam/emphasis).
# Collapsing all the way to 1 (not 2) gives the best chance of exactly
# matching the real word's spelling in the trained vocabulary, e.g.
# "stuuuupid" -> "stupid".
_RE_REPEATED_CHARS = re.compile(r"(.)\1{2,}")

# Single letter/digit/leet-symbol characters separated by one-or-more
# spaces/dots/dashes/underscores/asterisks, repeated 2+ times
# (e.g. "n.i.g.g.a", "s h i t", "f-u-c-k", "n.1.g.g.4") -> collapsed into one
# token by removing the separators, so leetspeak substitution can then run on
# the reassembled word.
_LEET_CHARS = r"a-zA-Z0-9@!$"
_RE_SPACED_LETTERS = re.compile(
    rf"\b(?:[{_LEET_CHARS}][\s.\-_*]+){{2,}}[{_LEET_CHARS}]\b")


def _collapse_spaced_letters(match: "re.Match") -> str:
    return re.sub(r"[\s.\-_*]", "", match.group(0))


def _leet_substitute_token(token: str) -> str:
    """Apply leet substitution only within tokens that already contain at
    least one real letter - this stops standalone numbers like '100' or '3'
    (no letters at all) from being garbled into fake words like 'ioo'.

    A LEADING '@' is stripped outright rather than substituted to 'a' - real
    testing showed substituting it (e.g. "@something" -> "asomething") was
    producing garbled, out-of-vocabulary tokens that triggered false-positive
    detections via the character n-gram features, which happened far more
    often in practice than it caught genuine leading-@ evasion (e.g.
    "@sshole"). Since "@johndoe"-style @mentions are the overwhelmingly more
    common real-world pattern, this trade-off was reversed: mentions are
    now stripped cleanly again, and '@' occurring mid-word (a rarer, more
    deliberate evasion pattern, e.g. "a@@hole") is still substituted."""
    if token.startswith("@"):
        token = token.lstrip("@")
    if not any(c.isalpha() for c in token):
        return token
    return _LEET_PATTERN.sub(lambda m: _LEET_SUBS[m.group(0)], token)


def normalize_evasion(text: str) -> str:
    """Undo common obfuscation tricks BEFORE the main cleaning pipeline runs,
    so words like 'n1gg4' or 'n.i.g.g.a' survive as recognizable words
    instead of being destroyed by the punctuation/number stripping step.
    Expects lowercased input; run after .lower() and URL/HTML stripping."""
    text = _RE_SPACED_LETTERS.sub(_collapse_spaced_letters, text)
    text = re.sub(r"\S+", lambda m: _leet_substitute_token(m.group(0)), text)
    text = _RE_REPEATED_CHARS.sub(r"\1", text)
    return text


def clean_text(text: str) -> str:
    """Apply the full cleaning + normalization pipeline to a single string."""
    text = str(text).lower().replace("’", "'")
    text = _RE_URL.sub(" ", text)
    text = _RE_HTML.sub(" ", text)
    text = _RE_CONTRACTION.sub(lambda m: _CONTRACTIONS[m.group(1)], text)
    text = normalize_evasion(text)
    text = text.replace("#", " ")
    text = _RE_NONALPHA.sub(" ", text)          # keep letters only
    text = _RE_SPACES.sub(" ", text).strip()

    # tokenize
    if _USE_NLTK:
        try:
            tokens = word_tokenize(text)
        except Exception:
            tokens = text.split()
    else:
        tokens = text.split()

    # remove stopwords + very short tokens, then lemmatize
    out = []
    for tok in tokens:
        if len(tok) < 2 or tok in _STOPWORDS:
            continue
        if _LEMMATIZER is not None:
            tok = _LEMMATIZER.lemmatize(tok)
        out.append(tok)
    return " ".join(out)


def clean_text_steps(text: str) -> dict:
    """
    Same pipeline as clean_text(), but returns every intermediate stage as a
    dict - used by the Data Preprocessing page to show a live before/after
    demo. Mirrors clean_text() exactly so the demo can never drift out of
    sync with what the models actually use.
    """
    original = str(text)
    steps = {"0. Original": original}

    s = original.lower().replace("’", "'")
    steps["1. Lowercase"] = s

    s = _RE_CONTRACTION.sub(lambda m: _CONTRACTIONS[m.group(1)], s)
    steps["1b. Expand contractions / preserve negation"] = s

    s = _RE_URL.sub(" ", s)
    s = _RE_HTML.sub(" ", s)
    steps["2. Remove URLs / HTML"] = _RE_SPACES.sub(" ", s).strip()

    s = normalize_evasion(s)
    steps["3. Evasion normalization (leetspeak, spaced letters, repeats)"] = \
        _RE_SPACES.sub(" ", s).strip()

    s = s.replace("#", " ")
    s = _RE_NONALPHA.sub(" ", s)
    s = _RE_SPACES.sub(" ", s).strip()
    steps["4. Remove remaining punctuation / numbers / special characters"] = s

    if _USE_NLTK:
        try:
            tokens = word_tokenize(s)
        except Exception:
            tokens = s.split()
    else:
        tokens = s.split()
    steps["5. Tokenization"] = " | ".join(tokens) if tokens else "(empty)"

    no_stop = [t for t in tokens if len(t) >= 2 and t not in _STOPWORDS]
    steps["6. Stopword removal"] = " | ".join(no_stop) if no_stop else "(empty)"

    if _LEMMATIZER is not None:
        lemmatized = [_LEMMATIZER.lemmatize(t) for t in no_stop]
    else:
        lemmatized = no_stop
    steps["7. Lemmatization"] = " | ".join(lemmatized) if lemmatized else "(empty)"

    steps["8. Final cleaned text (fed to the model)"] = " ".join(lemmatized)
    return steps


def preprocess_series(series):
    """Vectorized helper: clean an entire pandas Series of texts."""
    return series.astype(str).apply(clean_text)


if __name__ == "__main__":
    samples = [
        "You are SO stupid!!! @idiot check http://x.com #loser",
        "Have a great day everyone :)",
        "n1gg4 you are worthless",
        "n.i.g.g.a get out",
        "y o u   a r e   s t u p i d",
        "@sshole go away",
    ]
    for s in samples:
        print(f"{s!r}\n  -> {clean_text(s)!r}\n")

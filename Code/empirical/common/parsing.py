"""
Lightweight parsers for common LLM answer formats.
"""
from __future__ import annotations

import re
from typing import Optional

_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def parse_float(text: str) -> Optional[float]:
    """Return the first numeric token in the response, or None."""
    if text is None:
        return None
    m = _NUM_RE.search(text)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def parse_binary(text: str, true_words=("yes", "1", "true", "click", "would click"),
                 false_words=("no", "0", "false", "not click")) -> Optional[float]:
    """Return 1.0 / 0.0 / None from a yes/no-flavoured answer."""
    if text is None:
        return None
    t = text.strip().lower()
    for w in true_words:
        if t.startswith(w) or f" {w} " in f" {t} ":
            return 1.0
    for w in false_words:
        if t.startswith(w) or f" {w} " in f" {t} ":
            return 0.0
    return parse_float(t)  # fall back to a leading number


def parse_bounded(text: str, lo: float, hi: float) -> Optional[float]:
    """Parse a float and clip to [lo, hi]; None if no number found."""
    v = parse_float(text)
    if v is None:
        return None
    return float(max(lo, min(hi, v)))

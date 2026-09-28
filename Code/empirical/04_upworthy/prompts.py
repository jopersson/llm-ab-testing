"""
Prompt templates and output parser for the Upworthy CTR prediction task.

Design notes:
  - The system prompt fixes the setting (Upworthy / 2013-15 / Facebook feed)
    and the output format, but **does not** reveal the empirical CTR
    distribution. Leaking the distribution would make Y* partially
    anchored to the target population's marginal, which undermines
    Assumption 1 (Prentice) — we want Y* to carry information about Y
    through the headline, not through empirical-distribution disclosure in
    the prompt.
  - The parser interprets the integer completion as basis points
    (probability = value / 10_000), matching the prompt instruction.
    Answers above 10_000 bp (> 100 %) are clipped to 1.0 rather than
    dropped, and the calibration is relied upon to soak up residual bias.

If you edit the system prompt, increment PROMPT_VERSION so the LLM cache is
invalidated automatically (the cache key hashes the message payload, so any
byte-level change already forces a re-spend — PROMPT_VERSION is kept only as
a human-readable audit trail in logs).
"""

from __future__ import annotations

import re
from typing import Optional


PROMPT_VERSION = 2  # bumped 2026-04-22: removed empirical-distribution leak

SYSTEM = (
    "You estimate click-through rates (CTRs) for headlines posted on "
    "Upworthy.com between 2013 and 2015. Upworthy's traffic came mostly "
    "from the Facebook news feed in that period. Given a candidate "
    "headline, predict the CTR that a typical Upworthy-era Facebook user "
    "would exhibit when shown it in their feed. Respond with a single "
    "integer giving the predicted CTR in basis points (1 bp = 0.01 %). "
    "For example, an answer of '80' means 0.80 %. Do not include any "
    "other text, units, or commentary. Only the integer."
)


def build_user_message(headline: str) -> str:
    return f"Headline: {headline}\n\nPredicted CTR (basis points):"


def build_messages(headline: str):
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": build_user_message(headline)},
    ]


_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def parse_ctr_bps(text: str) -> Optional[float]:
    """
    Parse a basis-point CTR completion into a probability in [0, 1], or
    None if no number is found.

    The system prompt instructs the model to return a single integer in
    basis points (1 bp = 0.01 %), so the parsed number is interpreted as
    basis points unconditionally: probability = value / 10_000. Negative
    answers are floored at 0 and implausible answers above 10_000 bp
    (> 100 %) are clipped to 1.0 rather than dropped, so the parse rate is
    unaffected and downstream calibration absorbs any residual.

    NOTE: a previous version applied a magnitude-based scale heuristic
    (values in (1, 100] read as percent, > 100 as basis points). That rule
    was non-monotonic at 100 and contradicted the basis-point instruction,
    inflating Y* by ~100x and scrambling its rank ordering. It has been
    removed in favour of the deterministic basis-point parse below.
    """
    if text is None:
        return None
    m = _NUM_RE.search(text)
    if m is None:
        return None
    try:
        v = float(m.group(0))
    except ValueError:
        return None
    prob = v / 10000.0
    return float(min(1.0, max(0.0, prob)))

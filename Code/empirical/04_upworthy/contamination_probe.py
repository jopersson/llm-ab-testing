"""
Memorisation probe for Upworthy headlines.

Question:
    Does the LLM's Y* signal reflect a PREDICTION from headline semantics,
    or has the LLM MEMORISED the exact 2013-2015 Upworthy headlines (and
    possibly their CTRs) from its training corpus?

Method (headline-completion probe):
    Draw a random sample of N headlines from the dataset. For each, split
    the headline into a prefix of the first ~4 tokens and a suffix. Ask the
    LLM to complete the headline given the prefix, WITHOUT mentioning
    Upworthy. Compare the completion to the true suffix using:
        - exact match
        - token-level F1 (bag-of-tokens)
        - character-level Levenshtein ratio

    A memorisation-dominant LLM will produce near-exact completions;
    a non-memorisation LLM will produce plausible alternative headlines
    with very low token overlap.

The probe is cheap (~300 extra completions on a random sample of N=300)
and reports a single number -- the fraction of headlines whose completion
matches beyond a chance threshold -- that we can cite in the paper.

Usage:
    OPENAI_API_KEY=sk-... python3 contamination_probe.py --n 300
Outputs:
    output/tables/uw_contamination_probe.json
    output/figures/uw_contamination_probe.pdf  (histogram of F1 scores)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib as mpl
mpl.rcParams.update({
    "font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "lines.linewidth": 1.0, "lines.markersize": 3.5,
})
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EMP = HERE.parent
sys.path.insert(0, str(EMP))

from common.llm_client import LLMConfig, LLMCache, LLMRequest, run_batch  # noqa: E402

DATA_DIR = EMP / "output" / "data"
CACHE_PATH = EMP / "output" / "cache" / "uw_contamination.jsonl"
_OUT = Path(__import__("os").environ.get("UW_OUT_DIR", str(EMP / "output")))
FIG_DIR = _OUT / "figures"
TAB_DIR = _OUT / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

PREFIX_TOKENS = 4

PROBE_SYSTEM = (
    "You are an expert copywriter. Given the first few words of a "
    "headline, propose the most likely continuation. Respond ONLY with "
    "the continuation text -- no commentary, no quotation marks, no "
    "leading words already given. Keep the continuation under 120 "
    "characters."
)


def _tokens(s: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9']+", (s or "").lower())


def _token_f1(pred: str, truth: str) -> float:
    p, t = _tokens(pred), _tokens(truth)
    if not p or not t:
        return 0.0
    ps, ts = set(p), set(t)
    tp = len(ps & ts)
    if tp == 0:
        return 0.0
    precision = tp / len(ps)
    recall = tp / len(ts)
    return 2 * precision * recall / (precision + recall)


def _char_ratio(a: str, b: str) -> float:
    """Normalised longest common subsequence ratio (cheap Levenshtein-ish)."""
    a = (a or "").strip().lower()
    b = (b or "").strip().lower()
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    prev = [0] * (len(a) + 1)
    for j in range(1, len(b) + 1):
        curr = [0] * (len(a) + 1)
        for i in range(1, len(a) + 1):
            if a[i - 1] == b[j - 1]:
                curr[i] = prev[i - 1] + 1
            else:
                curr[i] = max(curr[i - 1], prev[i])
        prev = curr
    lcs = prev[-1]
    return (2 * lcs) / (len(a) + len(b))


def _split_prefix_suffix(headline: str) -> tuple[str, str]:
    toks = headline.split(" ")
    if len(toks) <= PREFIX_TOKENS + 1:
        return headline, ""
    prefix = " ".join(toks[:PREFIX_TOKENS])
    suffix = " ".join(toks[PREFIX_TOKENS:])
    return prefix, suffix


def _build_request(unit_id: str, headline: str) -> tuple[LLMRequest, str]:
    prefix, suffix = _split_prefix_suffix(headline)
    msgs = [
        {"role": "system", "content": PROBE_SYSTEM},
        {"role": "user", "content": f"Headline so far: \"{prefix}\"\n\nContinuation:"},
    ]
    return LLMRequest(unit_id=unit_id, draw_idx=0, messages=msgs), suffix


def _identity_parser(text: str):
    return text.strip() if text else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    df = pd.read_parquet(DATA_DIR / "uw_experiments.parquet")
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(df), size=min(args.n, len(df)), replace=False)
    sample = df.iloc[idx].copy()

    requests, suffixes = [], {}
    for _, row in sample.iterrows():
        req, suf = _build_request(row["unit_id"], row["headline"])
        requests.append(req)
        suffixes[row["unit_id"]] = suf

    cfg = LLMConfig(model=args.model, temperature=args.temperature,
                    max_tokens=120)
    cache = LLMCache(CACHE_PATH)

    if args.dry_run:
        import os as _os
        import common.llm_client as lc
        original = lc._call_one
        prev_key = _os.environ.get("OPENAI_API_KEY")
        _os.environ["OPENAI_API_KEY"] = "sk-dummy-dry-run"
        lc._call_one = lambda *a, **k: "a plausible made-up continuation"
        try:
            results = run_batch(requests, cfg, cache, _identity_parser)
        finally:
            lc._call_one = original
            if prev_key is None:
                _os.environ.pop("OPENAI_API_KEY", None)
            else:
                _os.environ["OPENAI_API_KEY"] = prev_key
    else:
        results = run_batch(requests, cfg, cache, _identity_parser)

    scores = []
    for rec in results:
        uid = rec["unit_id"]
        pred = rec.get("value") or ""
        truth = suffixes[uid]
        if not truth:
            continue
        scores.append({
            "unit_id": uid,
            "truth_suffix": truth,
            "pred_suffix": pred,
            "token_f1": _token_f1(pred, truth),
            "char_ratio": _char_ratio(pred, truth),
            "exact_match": int(pred.strip().lower() == truth.strip().lower()),
        })

    sc = pd.DataFrame(scores)

    summary = {
        "n_probed": int(len(sc)),
        "mean_token_f1": float(sc["token_f1"].mean()),
        "median_token_f1": float(sc["token_f1"].median()),
        "frac_token_f1_above_0_5": float((sc["token_f1"] >= 0.5).mean()),
        "frac_token_f1_above_0_8": float((sc["token_f1"] >= 0.8).mean()),
        "mean_char_ratio": float(sc["char_ratio"].mean()),
        "frac_exact_match": float(sc["exact_match"].mean()),
        "interpretation": (
            "High mean_token_f1 or frac_token_f1_above_0_8 indicates the LLM "
            "recovers the Upworthy suffix well given a short prefix, which "
            "is evidence of headline memorisation. A non-memorising LLM "
            "should produce semantically plausible but token-overlap-low "
            "continuations (token_f1 ~ 0.1-0.3)."
        ),
    }

    (TAB_DIR / "uw_contamination_probe.json").write_text(
        json.dumps(summary, indent=2))
    sc.to_parquet(TAB_DIR / "uw_contamination_probe_rows.parquet")

    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    # Carlini et al. (2023) non-memorized token-F1 range, drawn behind the bars.
    # Give the band a visible edge so its legend swatch is discernible.
    ax.axvspan(0.1, 0.3, facecolor="#999999", alpha=0.15,
               edgecolor="#777777", linewidth=0.8,
               label="Carlini et al. (2023)")
    ax.hist(sc["token_f1"], bins=25, color="#3366aa", edgecolor="white")
    ax.axvline(summary["mean_token_f1"], color="#cc3333", lw=1.5,
               label=f"mean = {summary['mean_token_f1']:.3f}")
    ax.set_xlabel("Token F1 (LLM continuation vs true suffix)")
    ax.set_ylabel("Count")
    ax.margins(y=0.20)
    # Opaque white frame so the grey band swatch is readable on top of the band.
    ax.legend(loc="upper right", frameon=True, framealpha=1.0,
              facecolor="white", edgecolor="0.7", borderaxespad=0.6,
              handlelength=1.4, handletextpad=0.5)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "uw_contamination_probe.pdf")
    fig.savefig(FIG_DIR / "uw_contamination_probe.png", dpi=180)
    plt.close(fig)

    print(json.dumps(summary, indent=2))
    print(f"wrote {TAB_DIR / 'uw_contamination_probe.json'}")
    print(f"wrote {FIG_DIR / 'uw_contamination_probe.pdf'}")


if __name__ == "__main__":
    main()

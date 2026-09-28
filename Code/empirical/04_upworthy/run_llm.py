"""
For each retained variant, query the LLM K times for a CTR estimate; cache
completions; merge back into a wide parquet with Ystar_0 ... Ystar_{K-1}.

Usage:
    OPENAI_API_KEY=sk-... python3 run_llm.py --K 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EMP = HERE.parent
sys.path.insert(0, str(EMP))

from common.llm_client import LLMConfig, LLMCache, LLMRequest, run_batch  # noqa: E402
from prompts import build_messages, parse_ctr_bps  # noqa: E402


DATA_DIR = EMP / "output" / "data"
CACHE_PATH = EMP / "output" / "cache" / "uw.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--max-variants", type=int, default=None)
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--seed-base", type=int, default=20260420)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    units = pd.read_parquet(DATA_DIR / "uw_experiments.parquet")
    if args.max_variants is not None:
        units = units.head(args.max_variants).copy()

    print(f"[uw] {len(units)} variants x K={args.K} draws "
          f"= {len(units) * args.K} completions")

    cfg = LLMConfig(model=args.model, temperature=args.temperature,
                    concurrency=args.concurrency, seed_base=args.seed_base)
    cache = LLMCache(CACHE_PATH)

    requests = []
    for _, row in units.iterrows():
        for k in range(args.K):
            requests.append(LLMRequest(
                unit_id=row["unit_id"],
                draw_idx=k,
                messages=build_messages(row["headline"]),
            ))

    if args.dry_run:
        import os as _os  # noqa: PLC0415
        import common.llm_client as lc  # noqa: PLC0415
        original = lc._call_one

        def refuser(client, cfg, req):
            raise RuntimeError("dry-run: no API call allowed")

        prev_key = _os.environ.get("OPENAI_API_KEY")
        _os.environ["OPENAI_API_KEY"] = "sk-dummy-dry-run"
        lc._call_one = refuser
        try:
            results = run_batch(requests, cfg, cache, parse_ctr_bps)
        finally:
            lc._call_one = original
            if prev_key is None:
                _os.environ.pop("OPENAI_API_KEY", None)
            else:
                _os.environ["OPENAI_API_KEY"] = prev_key
    else:
        results = run_batch(requests, cfg, cache, parse_ctr_bps)

    rec = pd.DataFrame(results)
    wide = rec.pivot(index="unit_id", columns="draw_idx", values="value")
    wide.columns = [f"Ystar_{k}" for k in wide.columns]
    wide = wide.reset_index()

    merged = units.merge(wide, on="unit_id", how="left")

    out_path = DATA_DIR / "uw_with_ystar.parquet"
    merged.to_parquet(out_path)
    print(f"[uw] wrote {out_path}  shape={merged.shape}")

    ystar_cols = [c for c in merged.columns if c.startswith("Ystar_")]
    parse_rate = (
        merged[ystar_cols].notna().sum().sum()
        / (len(merged) * len(ystar_cols))
    )
    print(f"[uw] parse rate: {parse_rate:.3%}")
    print(f"[uw] Ystar mean (across draws and variants): "
          f"{merged[ystar_cols].stack().mean():.4f}")


if __name__ == "__main__":
    main()

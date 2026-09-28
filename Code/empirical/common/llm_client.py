"""
Unified OpenAI chat-completion client with on-disk caching and bounded concurrency.

Design goals
------------
* Idempotent: a given (model, messages, temperature, seed, draw_idx) maps to
  exactly one completion cached on disk. Re-running the same job reads from
  cache and issues no new API calls.
* Resumable: crashes, rate-limit errors, or Ctrl+C only lose the in-flight batch.
* Parseable: callers pass a `parser(text) -> float | None`; unparseable outputs
  are recorded as NaN but still cached so we do not re-query them.
* Cheap to inspect: cache entries are JSONL in `output/cache/<name>.jsonl`.

The caller is responsible for setting OPENAI_API_KEY in the environment.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional

try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except ImportError:  # pragma: no cover
    _OPENAI_AVAILABLE = False


def _hash_key(payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


@dataclass
class LLMConfig:
    model: str = "gpt-4o-mini"
    temperature: float = 0.7
    max_tokens: int = 256
    timeout_s: float = 60.0
    max_retries: int = 6
    concurrency: int = 8
    seed_base: int = 20260420


@dataclass
class LLMRequest:
    unit_id: str
    draw_idx: int
    messages: list
    metadata: dict = field(default_factory=dict)


class LLMCache:
    """Append-only JSONL cache. One line per (request_key) completion."""

    def __init__(self, cache_path: Path):
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._mem: dict[str, dict] = {}
        self._load()

    def _load(self):
        if not self.cache_path.exists():
            return
        with self.cache_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                self._mem[rec["key"]] = rec

    def get(self, key: str) -> Optional[dict]:
        return self._mem.get(key)

    def put(self, key: str, record: dict):
        self._mem[key] = record
        with self.cache_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _request_key(cfg: LLMConfig, req: LLMRequest) -> str:
    return _hash_key({
        "model": cfg.model,
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens,
        "messages": req.messages,
        "draw_idx": req.draw_idx,
        "seed_base": cfg.seed_base,
    })


def _call_one(client, cfg: LLMConfig, req: LLMRequest) -> str:
    """Issue one completion with exponential-backoff retry."""
    last_err: Optional[Exception] = None
    for attempt in range(cfg.max_retries):
        try:
            resp = client.chat.completions.create(
                model=cfg.model,
                messages=req.messages,
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
                timeout=cfg.timeout_s,
                seed=cfg.seed_base + req.draw_idx,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:  # noqa: BLE001 — retry everything
            last_err = e
            wait = min(60.0, 2 ** attempt + 0.3 * attempt)
            time.sleep(wait)
    raise RuntimeError(f"LLM call failed after {cfg.max_retries} attempts: {last_err}")


def run_batch(
    requests: Iterable[LLMRequest],
    cfg: LLMConfig,
    cache: LLMCache,
    parser: Callable[[str], Optional[float]],
    verbose: bool = True,
) -> list[dict]:
    """
    Execute a batch of requests with cache-first semantics.

    Returns a list of records (one per request, in input order) with fields:
        unit_id, draw_idx, text, value, cached, metadata.
    Cached records skip the API entirely.
    """
    requests = list(requests)
    out: list[Optional[dict]] = [None] * len(requests)
    pending: list[tuple[int, LLMRequest, str]] = []

    for i, req in enumerate(requests):
        key = _request_key(cfg, req)
        hit = cache.get(key)
        if hit is not None:
            # Re-parse the cached raw completion rather than trusting the
            # stored value, so that a corrected/updated parser takes effect
            # without re-querying the API (parsing is deterministic and free).
            text = hit.get("text", "") or ""
            value = parser(text) if text and not text.startswith("[ERROR]") else None
            out[i] = {
                "unit_id": req.unit_id,
                "draw_idx": req.draw_idx,
                "text": text,
                "value": value,
                "cached": True,
                "metadata": req.metadata,
            }
        else:
            pending.append((i, req, key))

    if verbose:
        n_total = len(requests)
        print(f"[llm] {n_total - len(pending)}/{n_total} cached, "
              f"{len(pending)} to fetch "
              f"(model={cfg.model}, T={cfg.temperature}, c={cfg.concurrency})")

    if not pending:
        return [r for r in out if r is not None]

    if not _OPENAI_AVAILABLE:
        raise RuntimeError("openai package not installed. `pip install openai>=1.0`")
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set in the environment.")

    client = OpenAI()
    n_done = 0

    with ThreadPoolExecutor(max_workers=cfg.concurrency) as pool:
        futs = {pool.submit(_call_one, client, cfg, req): (i, req, key)
                for (i, req, key) in pending}
        for fut in as_completed(futs):
            i, req, key = futs[fut]
            try:
                text = fut.result()
            except Exception as e:  # noqa: BLE001
                text = f"[ERROR] {e}"
            value = parser(text) if not text.startswith("[ERROR]") else None
            cache.put(key, {
                "key": key, "text": text, "value": value,
                "unit_id": req.unit_id, "draw_idx": req.draw_idx,
            })
            out[i] = {
                "unit_id": req.unit_id,
                "draw_idx": req.draw_idx,
                "text": text,
                "value": value,
                "cached": False,
                "metadata": req.metadata,
            }
            n_done += 1
            if verbose and (n_done % 25 == 0 or n_done == len(pending)):
                print(f"[llm]   {n_done}/{len(pending)} fetched")

    # At this point every slot is filled.
    return [r for r in out if r is not None]

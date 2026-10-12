"""Retrieval evaluation across modes: Recall@k, MRR@10, doc hit@k, per-source, latency.

python -m medguide.eval.retrieval_eval [--modes dense bm25 hybrid hybrid_rerank] [--limit 10]
No LLM calls (uses the pre-generated questions).
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from medguide.config import load_config
from medguide.eval.common import UNVERIFIED, load_eval_questions, results_dir
from medguide.retrieval.pipeline import MODES, Retriever

KS = (1, 3, 5, 10)
MAX_K = 10


def rank_of(target: str, ids: list[str]) -> int | None:
    """1-based rank of target in ids, or None."""
    return ids.index(target) + 1 if target in ids else None


def item_metrics(gold_chunk: str, gold_doc: str, chunk_ids: list[str], doc_ids: list[str]) -> dict:
    r = rank_of(gold_chunk, chunk_ids)
    rd = rank_of(gold_doc, doc_ids)
    out = {"rank": r, "rr@10": 1.0 / r if r and r <= 10 else 0.0}
    for k in KS:
        out[f"recall@{k}"] = float(bool(r and r <= k))
        out[f"doc_hit@{k}"] = float(bool(rd and rd <= k))
    return out


def summarize(df: pd.DataFrame) -> dict:
    out = {"n": len(df)}
    for k in KS:
        out[f"recall@{k}"] = df[f"recall@{k}"].mean()
    out["mrr@10"] = df["rr@10"].mean()
    for k in (1, 5):
        out[f"doc_hit@{k}"] = df[f"doc_hit@{k}"].mean()
    out["latency_ms_p50"] = df["latency_ms"].median()
    out["latency_ms_p95"] = df["latency_ms"].quantile(0.95)
    return {
        k: (round(float(v), 4) if isinstance(v, float | np.floating) else v) for k, v in out.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", nargs="+", default=list(MODES))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config()
    questions, verified = load_eval_questions(cfg)
    if args.limit:
        questions = questions[: args.limit]
    label = "verified" if verified else UNVERIFIED
    print(f"{len(questions)} questions [{label}]")

    retriever = Retriever(cfg)
    rows = []
    for mode in args.modes:
        retriever.warmup(mode)
        retriever.retrieve("warm up query", mode=mode, top_k=MAX_K)
        for q in questions:
            t0 = time.perf_counter()
            hits = retriever.retrieve(q["question"], mode=mode, top_k=MAX_K)
            ms = (time.perf_counter() - t0) * 1000
            chunk_ids = [h.chunk["chunk_id"] for h in hits]
            doc_ids = list(dict.fromkeys(h.chunk["doc_id"] for h in hits))
            rows.append(
                {
                    "id": q["id"],
                    "mode": mode,
                    "source": q["source"],
                    "latency_ms": ms,
                    **item_metrics(q["gold_chunk_id"], q["gold_doc_id"], chunk_ids, doc_ids),
                }
            )
        print(f"  {mode}: done")

    per_item = pd.DataFrame(rows)
    summary = pd.DataFrame(
        [{"mode": m, **summarize(g)} for m, g in per_item.groupby("mode", sort=False)]
    )
    by_source = pd.DataFrame(
        [
            {
                "mode": m,
                "source": s,
                "n": len(g),
                "recall@5": round(g["recall@5"].mean(), 4),
                "mrr@10": round(g["rr@10"].mean(), 4),
            }
            for (m, s), g in per_item.groupby(["mode", "source"], sort=False)
        ]
    )

    suffix = "_smoke" if args.limit else ""
    out = results_dir(cfg)
    per_item.to_csv(out / f"retrieval_per_item{suffix}.csv", index=False)
    summary.to_csv(out / f"retrieval_summary{suffix}.csv", index=False)
    by_source.to_csv(out / f"retrieval_by_source{suffix}.csv", index=False)
    meta = {
        "label": label,
        "n_questions": len(questions),
        "embedding": cfg["embedding"]["model"],
        "reranker": cfg["reranker"]["model"],
    }
    (out / f"retrieval_meta{suffix}.json").write_text(json.dumps(meta, indent=2))

    print(f"\nRetrieval results [{label}], n={len(questions)}")
    print(summary.to_string(index=False))
    print("\nRecall@5 by source")
    print(by_source.pivot(index="mode", columns="source", values="recall@5").to_string())


if __name__ == "__main__":
    main()

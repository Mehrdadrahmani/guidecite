"""Run 5 hand-written queries through all four retrieval modes and print the top 3."""

from __future__ import annotations

import time

from medguide.config import load_config
from medguide.retrieval.pipeline import MODES, Retriever

QUERIES = [
    "What is the recommended first-line treatment for hypertension in adults?",
    "How long should a person with measles be isolated?",
    "Which vaccines are recommended during pregnancy?",
    "When should antibiotics be prescribed for a sore throat?",
    "What are the screening recommendations for tuberculosis in people with HIV?",
]


def main() -> None:
    cfg = load_config()
    retriever = Retriever(cfg)
    for mode in MODES:
        retriever.warmup(mode)
    for q in QUERIES:
        print("=" * 100 + f"\nQ: {q}")
        for mode in MODES:
            t0 = time.perf_counter()
            hits = retriever.retrieve(q, mode=mode, top_k=3)
            ms = (time.perf_counter() - t0) * 1000
            print(f"  [{mode}] {ms:.0f} ms")
            for h in hits:
                c = h.chunk
                print(
                    f"    {h.score:8.4f}  {c['source'].upper():4s} {c['title'][:60]!r}"
                    f" | {str(c['section'])[:40]!r}"
                )


if __name__ == "__main__":
    main()

"""Sparse retrieval over the persisted BM25 index."""

from __future__ import annotations

from medguide.index.bm25 import BM25Index


def bm25_search(index: BM25Index, query: str, k: int) -> list[tuple[dict, float]]:
    return index.search(query, k)

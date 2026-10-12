"""Reciprocal rank fusion."""

from __future__ import annotations

from collections.abc import Sequence


def rrf_fuse(rankings: Sequence[Sequence[str]], k: int = 60) -> list[tuple[str, float]]:
    """Fuse ranked id lists: score(d) = sum over lists of 1 / (k + rank), rank starting at 1.

    Ties are broken by first appearance, which keeps the output deterministic.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: -kv[1])

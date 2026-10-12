"""Dense retrieval: embed the query, nearest neighbours in Qdrant (cosine)."""

from __future__ import annotations

from medguide.index.qdrant_store import search


def dense_search(embedder, client, collection: str, query: str, k: int) -> list[tuple[dict, float]]:
    return search(client, collection, embedder.embed_query(query), k)

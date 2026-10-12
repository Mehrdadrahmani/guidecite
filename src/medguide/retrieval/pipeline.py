"""One retrieval interface over four modes: dense | bm25 | hybrid | hybrid_rerank."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from medguide.retrieval.bm25 import bm25_search
from medguide.retrieval.dense import dense_search
from medguide.retrieval.hybrid import rrf_fuse

MODES = ("dense", "bm25", "hybrid", "hybrid_rerank")


@dataclass
class Hit:
    chunk: dict
    score: float


@dataclass
class Retriever:
    """Loads components lazily so e.g. `bm25` mode never loads the embedding model."""

    cfg: dict
    _embedder: object = field(default=None, repr=False)
    _client: object = field(default=None, repr=False)
    _bm25: object = field(default=None, repr=False)
    _reranker: object = field(default=None, repr=False)

    # --- components -------------------------------------------------------
    @property
    def embedder(self):
        if self._embedder is None:
            from medguide.index.embed import load_embedder
            from medguide.index.qdrant_store import read_manifest

            # query with the same model the index was built with
            self._embedder = load_embedder(self.cfg, read_manifest(self.cfg)["embedding_model"])
        return self._embedder

    @property
    def client(self):
        if self._client is None:
            from medguide.index.qdrant_store import get_client

            self._client = get_client(self.cfg)
        return self._client

    @property
    def bm25(self):
        if self._bm25 is None:
            from medguide.index.bm25 import BM25Index

            path = Path(self.cfg["paths"]["index_dir"]) / self.cfg["bm25"]["filename"]
            self._bm25 = BM25Index.load(path)
        return self._bm25

    @property
    def reranker(self):
        if self._reranker is None:
            from medguide.config import detect_device
            from medguide.retrieval.rerank import CrossEncoderReranker

            rcfg = self.cfg["reranker"]
            self._reranker = CrossEncoderReranker(
                rcfg["model"], detect_device(), rcfg["batch_size"]
            )
        return self._reranker

    def warmup(self, mode: str) -> None:
        if mode in ("dense", "hybrid", "hybrid_rerank"):
            _ = self.embedder, self.client
        if mode in ("bm25", "hybrid", "hybrid_rerank"):
            _ = self.bm25
        if mode == "hybrid_rerank":
            _ = self.reranker

    # --- retrieval --------------------------------------------------------
    def dense(self, query: str, k: int) -> list[Hit]:
        collection = self.cfg["vector_store"]["collection"]
        res = dense_search(self.embedder, self.client, collection, query, k)
        return [Hit(c, s) for c, s in res]

    def sparse(self, query: str, k: int) -> list[Hit]:
        return [Hit(c, s) for c, s in bm25_search(self.bm25, query, k)]

    def hybrid(self, query: str, k: int) -> list[Hit]:
        rcfg = self.cfg["retrieval"]
        dense = self.dense(query, rcfg["dense_k"])
        sparse = self.sparse(query, rcfg["bm25_k"])
        by_id = {h.chunk["chunk_id"]: h.chunk for h in dense + sparse}
        fused = rrf_fuse(
            [[h.chunk["chunk_id"] for h in dense], [h.chunk["chunk_id"] for h in sparse]],
            k=rcfg["rrf_k"],
        )
        return [Hit(by_id[cid], score) for cid, score in fused[:k]]

    def retrieve(self, query: str, mode: str | None = None, top_k: int | None = None) -> list[Hit]:
        rcfg = self.cfg["retrieval"]
        mode = mode or rcfg["mode"]
        top_k = top_k or rcfg["top_k"]
        if mode == "dense":
            return self.dense(query, top_k)
        if mode == "bm25":
            return self.sparse(query, top_k)
        if mode == "hybrid":
            return self.hybrid(query, top_k)
        if mode == "hybrid_rerank":
            candidates = self.hybrid(query, rcfg["rerank_candidates"])
            ranked = self.reranker.rerank(query, [h.chunk for h in candidates], top_k)
            return [Hit(c, s) for c, s in ranked]
        raise ValueError(f"Unknown retrieval mode '{mode}'. Choose one of {MODES}.")

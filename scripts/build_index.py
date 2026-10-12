"""Embed chunks into Qdrant and build the BM25 index.

Usage: python scripts/build_index.py [--config ...]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from medguide.config import load_config
from medguide.index.bm25 import BM25Index
from medguide.index.embed import load_embedder
from medguide.index.qdrant_store import build_collection, get_client, write_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)

    chunks_path = Path(cfg["paths"]["processed_dir"]) / "chunks.parquet"
    if not chunks_path.exists():
        raise SystemExit(f"{chunks_path} not found. Run `make data` first.")
    chunks = pd.read_parquet(chunks_path)
    records = [
        {k: (None if pd.isna(v) else v) for k, v in r.items()} for r in chunks.to_dict("records")
    ]
    print(f"{len(records)} chunks")

    embedder = load_embedder(cfg)
    print(
        f"embedding with {embedder.model_name} on {embedder.model.device}, batch {embedder.batch_size}"
    )
    t0 = time.perf_counter()
    vectors = embedder.embed_documents([r["text"] for r in records])
    print(f"embedded in {time.perf_counter() - t0:.1f}s, dim={len(vectors[0])}")

    client = get_client(cfg)
    collection = cfg["vector_store"]["collection"]
    build_collection(client, collection, records, vectors)
    print(
        f"qdrant collection '{collection}' ({cfg['vector_store']['mode']}): "
        f"{client.count(collection).count} points"
    )
    client.close()

    bm25_path = Path(cfg["paths"]["index_dir"]) / cfg["bm25"]["filename"]
    BM25Index(records, language=cfg["language"]).save(bm25_path)
    print(f"bm25 index saved to {bm25_path}")

    write_manifest(
        cfg,
        {
            "embedding_model": embedder.model_name,
            "dim": len(vectors[0]),
            "n_chunks": len(records),
            "collection": collection,
            "language": cfg["language"],
        },
    )


if __name__ == "__main__":
    main()

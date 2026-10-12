"""Download epfl-llm/guidelines, keep WHO/CDC/NICE, sample, chunk, and save parquet.

Usage: python scripts/download_data.py [--config configs/default.yaml]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from medguide.config import load_config
from medguide.data.chunk import chunk_corpus, chunk_stats
from medguide.data.clean import clean_documents, filter_and_sample
from medguide.data.load import load_guidelines


def validate_chunks(chunks) -> None:
    required = ["doc_id", "source", "title", "chunk_id", "chunk_index", "text"]
    assert chunks["text"].str.strip().str.len().gt(0).all(), "empty chunk found"
    for col in required:
        assert chunks[col].notna().all(), f"missing metadata: {col}"
    assert chunks["chunk_id"].is_unique, "duplicate chunk_id"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    out_dir = Path(cfg["paths"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = load_guidelines(cfg)
    print(f"loaded {len(raw)} docs from sources {cfg['data']['sources']}")
    print(raw["source"].value_counts().to_string())
    docs = filter_and_sample(clean_documents(raw), cfg)
    print(f"\nsampled {len(docs)} docs:\n{docs['source'].value_counts().to_string()}")
    print(f"words per doc: median={docs['n_words'].median():.0f}, max={docs['n_words'].max()}")
    docs.to_parquet(out_dir / "documents.parquet", index=False)

    chunks = chunk_corpus(docs, cfg)
    validate_chunks(chunks)
    chunks.to_parquet(out_dir / "chunks.parquet", index=False)

    stats = chunk_stats(chunks)
    stats["docs_per_source"] = docs["source"].value_counts().to_dict()
    stats["docs_without_url"] = int(docs["url"].isna().sum())
    stats["chunks_with_section"] = int(chunks["section"].notna().sum())
    print("\nchunk stats:\n" + json.dumps(stats, indent=2))
    results = Path(cfg["paths"]["results_dir"])
    results.mkdir(parents=True, exist_ok=True)
    (results / "data_stats.json").write_text(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()

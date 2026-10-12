"""Load the epfl-llm/guidelines dataset and keep the configured sources."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

KEEP_COLUMNS = ["id", "source", "title", "url", "overview"]
RAW_FILE = "open_guidelines.jsonl"


def download_raw(cfg: dict) -> Path:
    """Fetch the raw JSONL from the Hub into data/raw (no-op if already present)."""
    from huggingface_hub import hf_hub_download

    raw_dir = Path(cfg["paths"]["raw_dir"])
    target = raw_dir / RAW_FILE
    if target.exists():
        return target
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(
        repo_id=cfg["data"]["hf_dataset"],
        filename=RAW_FILE,
        repo_type="dataset",
        local_dir=raw_dir,
    )
    return Path(path)


def load_guidelines(cfg: dict) -> pd.DataFrame:
    """Return guideline documents from the configured sources.

    Columns: id, source, title, url, overview, text.
    """
    dcfg = cfg["data"]
    path = download_raw(cfg)
    sources = set(dcfg["sources"])
    cols = KEEP_COLUMNS + [dcfg["text_field"]]
    parts = []
    # stream in chunks: the file is ~880 MB, but only ~2.5k rows are kept
    for frame in pd.read_json(path, lines=True, chunksize=5000, dtype=False):
        parts.append(frame.loc[frame["source"].isin(sources), cols])
    df = pd.concat(parts).rename(columns={dcfg["text_field"]: "text"})
    return df.reset_index(drop=True)

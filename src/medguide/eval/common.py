"""Shared helpers for evaluation scripts."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

UNVERIFIED = "UNVERIFIED (synthetic)"


def load_chunks(cfg: dict) -> pd.DataFrame:
    path = Path(cfg["paths"]["processed_dir"]) / "chunks.parquet"
    if not path.exists():
        raise SystemExit(f"{path} not found. Run `make data` first.")
    df = pd.read_parquet(path)
    return df.astype(object).where(df.notna(), None)


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def print_estimate(task: str, n_calls: int, model: str) -> None:
    print(f"[cost] {task}: ~{n_calls} LLM calls with {model}")


def batch_invoke(llm, prompts: list, max_concurrency: int) -> list:
    """Run prompts concurrently; failed calls come back as exceptions instead of aborting."""
    return llm.batch(prompts, config={"max_concurrency": max_concurrency}, return_exceptions=True)


def results_dir(cfg: dict) -> Path:
    path = Path(cfg["paths"]["results_dir"])
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_eval_questions(cfg: dict) -> tuple[list[dict], bool]:
    """Prefer hand-verified questions; else the kept draft questions. Returns (rows, verified)."""
    eval_dir = Path(cfg["paths"]["eval_dir"])
    verified = eval_dir / "questions_verified.jsonl"
    if verified.exists():
        return read_jsonl(verified), True
    draft = eval_dir / "questions_draft.jsonl"
    if not draft.exists():
        raise SystemExit(f"{draft} not found. Run `make eval-questions` first.")
    return [r for r in read_jsonl(draft) if r.get("keep", True)], False

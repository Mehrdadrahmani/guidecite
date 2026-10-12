"""Generate paraphrased, single-chunk-answerable questions for retrieval evaluation.

python -m medguide.eval.build_questions [--limit 10] [--n 200]
Writes eval/questions_draft.jsonl and eval/review.csv (smoke runs write *_smoke files).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from medguide.config import load_config
from medguide.eval.common import (
    batch_invoke,
    enable_llm_cache,
    load_chunks,
    print_estimate,
    write_jsonl,
)
from medguide.generation.llm import (
    generator_from_config,
    judge_from_config,
    load_prompt,
    text_of,
)

OVERSAMPLE = 1.8  # extra candidates to absorb SKIP / DROP

# sections about how a guideline was made, not about care: no useful clinical questions
_ADMIN_SECTION = re.compile(
    r"method|development|grade|evidence (?:profile|summary|review)|search strateg|declaration|"
    r"interest|acknowledg|reference|contributor|member|dissemination|update|funding|"
    r"abbreviation|glossary|contents|annex|appendix|disclosure|continuing education",
    re.IGNORECASE,
)
# deterministic guard: small judges miss these
_NOT_SELF_CONTAINED = re.compile(
    r"\b(the|this|that) (passage|text|document|guideline|guidance|report|statement|study|"
    r"studies|section|excerpt)\b|according to",
    re.IGNORECASE,
)


def _body(text: str) -> str:
    return text.split("\n\n", 1)[-1] if text.startswith("Title:") else text


def sample_chunks(chunks: pd.DataFrame, n: int, sources: list[str], seed: int) -> pd.DataFrame:
    """Stratified by source; very short chunks are excluded (rarely contain a fact)."""
    pool = chunks[chunks["n_tokens"] >= 120]
    pool = pool[~pool["section"].fillna("").str.contains(_ADMIN_SECTION)]
    per_source = int(round(n * OVERSAMPLE / len(sources)))
    parts = [
        pool[pool["source"] == s].sample(
            n=min(per_source, (pool["source"] == s).sum()), random_state=seed
        )
        for s in sources
    ]
    return pd.concat(parts).reset_index(drop=True)


def clean_question(text: str) -> str | None:
    q = text.strip().strip('"').strip()
    if not q or q.upper().startswith("SKIP") or "?" not in q:
        return None
    q = q.splitlines()[0].strip()
    return None if _NOT_SELF_CONTAINED.search(q) else q


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=None, help="target number of questions")
    parser.add_argument("--limit", type=int, default=None, help="smoke test: N candidates only")
    args = parser.parse_args()
    cfg = load_config()
    print(f"LLM cache: {enable_llm_cache(cfg)}")
    ecfg = cfg["eval"]
    n_target = args.n or ecfg["n_questions"]
    sources = cfg["data"]["sources"]

    cands = sample_chunks(load_chunks(cfg), n_target, sources, cfg["seed"])
    if args.limit:
        cands = (
            cands.groupby("source")
            .head(-(-args.limit // len(sources)))
            .head(args.limit)
            .reset_index(drop=True)
        )
    print_estimate(
        "question generation + filter",
        2 * len(cands),
        f"{cfg['llm']['model']} / judge {cfg['judge']['model']}",
    )

    gen_tmpl = load_prompt(cfg, "question_gen.txt")
    gen = batch_invoke(
        generator_from_config(cfg),
        [[("user", gen_tmpl.format(passage=_body(t)))] for t in cands["text"]],
        ecfg["max_concurrency"],
    )
    questions = [None if isinstance(r, Exception) else clean_question(text_of(r)) for r in gen]
    n_errors = sum(isinstance(r, Exception) for r in gen)

    filt_tmpl = load_prompt(cfg, "question_filter.txt")
    idx = [i for i, q in enumerate(questions) if q]
    verdicts = batch_invoke(
        judge_from_config(cfg, max_tokens=8),
        [
            [("user", filt_tmpl.format(question=questions[i], passage=_body(cands.at[i, "text"])))]
            for i in idx
        ],
        ecfg["max_concurrency"],
    )
    keep = {
        i: (not isinstance(v, Exception) and "KEEP" in text_of(v).upper())
        for i, v in zip(idx, verdicts, strict=True)
    }

    rows = []
    for i, q in enumerate(questions):
        if not q or not keep.get(i):
            continue
        c = cands.iloc[i]
        rows.append(
            {
                "question": q,
                "gold_chunk_id": c["chunk_id"],
                "gold_doc_id": c["doc_id"],
                "source": c["source"],
                "title": c["title"],
                "keep": True,
            }
        )
    # trim to the per-source target while staying balanced
    per_source = -(-n_target // len(sources))
    df = pd.DataFrame(rows)
    if not args.limit and len(df):
        df = df.groupby("source", group_keys=False).head(per_source)
    df.insert(0, "id", [f"q{i:04d}" for i in range(len(df))])
    rows = df.to_dict("records")

    print(f"candidates={len(cands)} generated={len(idx)} kept={len(rows)} errors={n_errors}")
    if len(df):
        print(df["source"].value_counts().to_string())
    suffix = "_smoke" if args.limit else ""
    eval_dir = Path(cfg["paths"]["eval_dir"])
    write_jsonl(eval_dir / f"questions_draft{suffix}.jsonl", rows)

    text_by_id = dict(zip(cands["chunk_id"], cands["text"], strict=True))
    review = pd.DataFrame(
        {
            "id": [r["id"] for r in rows],
            "question": [r["question"] for r in rows],
            "gold_chunk_text": [_body(text_by_id[r["gold_chunk_id"]]) for r in rows],
            "keep": "",
            "edited_question": "",
        }
    )
    review.to_csv(eval_dir / f"review{suffix}.csv", index=False)
    print(f"wrote {eval_dir / f'questions_draft{suffix}.jsonl'} and review{suffix}.csv")
    for r in rows[:5]:
        print(f"  {r['source']}: {r['question']}")


if __name__ == "__main__":
    main()

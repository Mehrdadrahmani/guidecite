"""Generation quality: RAGAS, citation validity, and abstention on out-of-scope questions.

python -m medguide.eval.ragas_eval [--limit 10] [--mode hybrid_rerank] [--skip-ragas]

RAGAS has no reference answers here, so the gold chunk (the passage each synthetic question was
written from) is used as the `reference` for context precision / recall.
"""

from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from medguide.config import load_config
from medguide.eval.common import (
    UNVERIFIED,
    enable_llm_cache,
    load_chunks,
    load_eval_questions,
    print_estimate,
    read_jsonl,
    results_dir,
)
from medguide.generation.answer import AnswerEngine

_ABSTAIN = re.compile(
    r"do(es)? not contain enough information|not enough information|insufficient information|"
    r"cannot (be )?answer|can't answer|unable to answer|no information (about|on)|"
    r"do(es)? not (address|cover|mention|provide)",
    re.IGNORECASE,
)
_NOTE = re.compile(r"\n*\s*Note:.*not medical advice\.?\s*$", re.IGNORECASE | re.DOTALL)


def is_abstention(answer: str) -> bool:
    return bool(_ABSTAIN.search(answer))


def strip_disclaimer(answer: str) -> str:
    """The closing disclaimer is not a claim about the context; keep it out of faithfulness."""
    return _NOTE.sub("", answer).strip()


def balanced_subset(rows: list[dict], n: int) -> list[dict]:
    """Round-robin across sources so the subset stays balanced."""
    by_source: dict[str, list[dict]] = {}
    for r in rows:
        by_source.setdefault(r["source"], []).append(r)
    out: list[dict] = []
    while len(out) < n and any(by_source.values()):
        for items in by_source.values():
            if items and len(out) < n:
                out.append(items.pop(0))
    return out


def answer_all(engine: AnswerEngine, questions: list[str], mode: str, workers: int) -> list:
    def run(q: str):
        try:
            return engine.answer(q, mode=mode)
        except Exception as exc:  # keep going; failures are reported
            return exc

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(run, questions))


def run_ragas(cfg: dict, samples: list[dict], workers: int) -> pd.DataFrame:
    from ragas import EvaluationDataset, RunConfig, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics._answer_relevance import ResponseRelevancy
    from ragas.metrics._context_precision import LLMContextPrecisionWithReference
    from ragas.metrics._context_recall import LLMContextRecall
    from ragas.metrics._faithfulness import Faithfulness

    from medguide.generation.llm import judge_from_config
    from medguide.index.embed import load_embedder

    judge = LangchainLLMWrapper(judge_from_config(cfg, max_tokens=4096))
    embeddings = LangchainEmbeddingsWrapper(load_embedder(cfg))
    metrics = [
        Faithfulness(),
        ResponseRelevancy(strictness=1),  # 1 generated question per answer keeps calls down
        LLMContextPrecisionWithReference(),
        LLMContextRecall(),
    ]
    result = evaluate(
        EvaluationDataset.from_list(samples),
        metrics=metrics,
        llm=judge,
        embeddings=embeddings,
        run_config=RunConfig(max_workers=workers, timeout=900, max_retries=3),
        raise_exceptions=False,
    )
    return result.to_pandas()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--mode", default=None)
    parser.add_argument("--skip-ragas", action="store_true")
    args = parser.parse_args()
    cfg = load_config()
    print(f"LLM cache: {enable_llm_cache(cfg)}")
    ecfg = cfg["eval"]
    mode = args.mode or cfg["retrieval"]["mode"]
    workers = ecfg["max_concurrency"]

    questions, verified = load_eval_questions(cfg)
    subset = balanced_subset(questions, args.limit or ecfg["ragas_subset"])
    oos = read_jsonl(Path(cfg["paths"]["eval_dir"]) / "out_of_scope.jsonl")
    if args.limit:
        oos = oos[: max(2, args.limit // 2)]
    n_ragas_calls = 0 if args.skip_ragas else len(subset) * 8  # ~2 faithfulness + 1-2 others
    print_estimate(
        "answers (in-scope + out-of-scope)",
        len(subset) + len(oos),
        f"{cfg['llm']['provider']}:{cfg['llm']['model']}",
    )
    print_estimate(
        "RAGAS judge", n_ragas_calls, f"{cfg['judge']['provider']}:{cfg['judge']['model']}"
    )

    engine = AnswerEngine(cfg)
    engine.retriever.warmup(mode)
    answers = answer_all(engine, [q["question"] for q in subset], mode, workers)
    oos_answers = answer_all(engine, [q["question"] for q in oos], mode, workers)

    chunk_text = dict(zip(*[load_chunks(cfg)[c] for c in ("chunk_id", "text")], strict=True))
    rows, samples = [], []
    for q, out in zip(subset, answers, strict=True):
        if isinstance(out, Exception):
            rows.append({"id": q["id"], "source": q["source"], "error": str(out)})
            continue
        retrieved_ids = [r["chunk"]["chunk_id"] for r in out["retrieved"]]
        rows.append(
            {
                "id": q["id"],
                "source": q["source"],
                "question": q["question"],
                "answer": out["answer"],
                "n_citations": len(out["citations"]),
                "citations_valid": out["citations_valid"],
                "has_citation": bool(out["citations"]),
                "abstained": is_abstention(out["answer"]),
                "gold_retrieved": q["gold_chunk_id"] in retrieved_ids,
                "generation_ms": out["timings"]["generation_ms"],
                "error": "",
            }
        )
        samples.append(
            {
                "user_input": q["question"],
                "response": strip_disclaimer(out["answer"]),
                "retrieved_contexts": [r["chunk"]["text"] for r in out["retrieved"]],
                "reference": chunk_text[q["gold_chunk_id"]],
            }
        )
    per_item = pd.DataFrame(rows)

    oos_rows = []
    for q, out in zip(oos, oos_answers, strict=True):
        err = isinstance(out, Exception)
        oos_rows.append(
            {
                "id": q["id"],
                "type": q["type"],
                "question": q["question"],
                "answer": "" if err else out["answer"],
                "abstained": False if err else is_abstention(out["answer"]),
                "n_citations": 0 if err else len(out["citations"]),
                "error": str(out) if err else "",
            }
        )
    oos_df = pd.DataFrame(oos_rows)

    small_judge = cfg["judge"]["provider"] in cfg["judge"].get("small_judge_models", [])
    summary = {
        "label": "verified" if verified else UNVERIFIED,
        "mode": mode,
        "generator": f"{cfg['llm']['provider']}:{cfg['llm']['model']}",
        "judge": f"{cfg['judge']['provider']}:{cfg['judge']['model']}",
        "judge_reliability": "small-judge, low reliability" if small_judge else "standard",
        "same_family_judge": cfg["judge"]["model"] == cfg["llm"]["model"],
        "n_in_scope": len(subset),
        "n_errors": int((per_item["error"] != "").sum()),
    }
    ok = per_item[per_item["error"] == ""]
    if len(ok):
        summary.update(
            {
                "citation_validity_rate": round(float(ok["citations_valid"].mean()), 4),
                "answers_with_citation_rate": round(float(ok["has_citation"].mean()), 4),
                "in_scope_abstention_rate": round(float(ok["abstained"].mean()), 4),
                "gold_chunk_in_context_rate": round(float(ok["gold_retrieved"].mean()), 4),
                "generation_ms_p50": round(float(ok["generation_ms"].median()), 1),
            }
        )
    if len(oos_df):
        summary["out_of_scope"] = {
            "n": len(oos_df),
            "correct_abstention_rate": round(float(oos_df["abstained"].mean()), 4),
            "by_type": {
                t: round(float(g["abstained"].mean()), 4) for t, g in oos_df.groupby("type")
            },
        }

    if not args.skip_ragas and samples:
        scores = run_ragas(cfg, samples, workers)
        metric_cols = [
            c
            for c in scores.columns
            if c not in ("user_input", "response", "retrieved_contexts", "reference")
        ]
        scores.insert(0, "id", ok["id"].tolist())
        summary["ragas"] = {c: round(float(scores[c].mean()), 4) for c in metric_cols}
        summary["ragas_nan_counts"] = {c: int(scores[c].isna().sum()) for c in metric_cols}
        suffix = "_smoke" if args.limit else ""
        scores.drop(columns=["retrieved_contexts", "reference"]).to_csv(
            results_dir(cfg) / f"ragas_per_item{suffix}.csv", index=False
        )

    suffix = "_smoke" if args.limit else ""
    out_dir = results_dir(cfg)
    per_item.to_csv(out_dir / f"generation_per_item{suffix}.csv", index=False)
    oos_df.to_csv(out_dir / f"abstention{suffix}.csv", index=False)
    (out_dir / f"generation_summary{suffix}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

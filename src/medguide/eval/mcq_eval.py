"""MCQ benchmark: does retrieval from the guideline corpus help on exam questions?

python -m medguide.eval.mcq_eval [--limit 10] [--modes none dense hybrid hybrid_rerank]
Same LLM, same prompt, temperature 0 for every condition. Accuracy with Wilson 95% CI and an
exact McNemar test of each retrieval mode against no-retrieval on the same items.
"""

from __future__ import annotations

import argparse
import json
import re

import pandas as pd

from medguide.config import load_config
from medguide.eval.common import batch_invoke, enable_llm_cache, print_estimate, results_dir
from medguide.generation.answer import format_context
from medguide.generation.llm import generator_from_config, load_prompt, text_of
from medguide.retrieval.pipeline import Retriever

LETTERS = "ABCD"
CONDITIONS = ("none", "dense", "hybrid", "hybrid_rerank")
_FINAL = re.compile(r"answer\s*(?:is)?\s*[:\-]?\s*\(?\**\s*([A-D])\b", re.IGNORECASE)


def load_mcq(cfg: dict, n: int, seed: int) -> pd.DataFrame:
    """Sample n MedQA questions, stratified by exam step (meta_info)."""
    from datasets import load_dataset

    ecfg = cfg["eval"]
    df = load_dataset(ecfg["mcq_dataset"], split=ecfg["mcq_split"]).to_pandas()
    df = df[df["answer_idx"].isin(list(LETTERS))].reset_index(drop=True)
    frac = n / len(df)
    sample = (
        df.groupby("meta_info", group_keys=False)
        .apply(lambda g: g.sample(n=max(1, round(len(g) * frac)), random_state=seed))
        .head(n)
    )
    sample = sample.reset_index().rename(columns={"index": "row"})
    sample["id"] = [f"mcq{r:05d}" for r in sample["row"]]
    return sample


def format_options(options: dict) -> str:
    return "\n".join(f"{k}. {options[k]}" for k in LETTERS)


def parse_letter(text: str) -> str | None:
    """Robust answer extraction: last 'Answer: X', else a lone letter on the last line."""
    matches = _FINAL.findall(text)
    if matches:
        return matches[-1].upper()
    last = text.strip().splitlines()[-1] if text.strip() else ""
    m = re.fullmatch(r"\W*([A-D])\W*", last.strip())
    return m.group(1) if m else None


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    from statsmodels.stats.proportion import proportion_confint

    lo, hi = proportion_confint(k, n, alpha=0.05, method="wilson")
    return float(lo), float(hi)


def mcnemar_p(base: pd.Series, other: pd.Series) -> tuple[int, int, float]:
    """Exact McNemar on paired correctness. Returns (b, c, p): b = base right/other wrong."""
    from statsmodels.stats.contingency_tables import mcnemar

    b = int((base & ~other).sum())
    c = int((~base & other).sum())
    table = [[int((base & other).sum()), b], [c, int((~base & ~other).sum())]]
    return b, c, float(mcnemar(table, exact=True).pvalue)


def build_prompt(tmpl: str, row, passages: list[dict] | None) -> str:
    if passages:
        instr = (
            "Excerpts from clinical guidelines are provided; use them if relevant, "
            "otherwise rely on your medical knowledge."
        )
        context = "Guideline excerpts:\n<context>\n" + format_context(passages) + "\n</context>\n\n"
    else:
        instr, context = "", ""
    return tmpl.format(
        context_instruction=instr,
        context=context,
        question=row["question"],
        options=format_options(row["options"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--modes", nargs="+", default=list(CONDITIONS))
    args = parser.parse_args()
    cfg = load_config()
    print(f"LLM cache: {enable_llm_cache(cfg)}")
    ecfg = cfg["eval"]
    n = args.limit or ecfg["mcq_n"]
    mcq = load_mcq(cfg, n, cfg["seed"])
    model = f"{cfg['llm']['provider']}:{cfg['llm']['model']}"
    print_estimate("MCQ", len(mcq) * len(args.modes), model)

    tmpl = load_prompt(cfg, "mcq.txt")
    llm = generator_from_config(cfg)
    retriever = Retriever(cfg)
    top_k = cfg["retrieval"]["top_k"]
    rows = []
    for mode in args.modes:
        passages = [None] * len(mcq)
        if mode != "none":
            retriever.warmup(mode)
            # query = question stem + options, so option-specific drugs/conditions can match
            passages = [
                [
                    h.chunk
                    for h in retriever.retrieve(
                        f"{r['question']}\n{format_options(r['options'])}", mode=mode, top_k=top_k
                    )
                ]
                for _, r in mcq.iterrows()
            ]
        prompts = [
            [("user", build_prompt(tmpl, r, p))]
            for (_, r), p in zip(mcq.iterrows(), passages, strict=True)
        ]
        replies = batch_invoke(llm, prompts, ecfg["max_concurrency"])
        for (_, r), reply, p in zip(mcq.iterrows(), replies, passages, strict=True):
            err = isinstance(reply, Exception)
            text = "" if err else text_of(reply)
            pred = None if err else parse_letter(text)
            rows.append(
                {
                    "id": r["id"],
                    "mode": mode,
                    "step": r["meta_info"],
                    "gold": r["answer_idx"],
                    "pred": pred,
                    "correct": pred == r["answer_idx"],
                    "error": str(reply) if err else "",
                    "retrieved_sources": ";".join(c["source"] for c in p) if p else "",
                    "retrieved_titles": " | ".join(c["title"][:60] for c in p) if p else "",
                    "raw_tail": text[-200:],
                }
            )
        acc = sum(x["correct"] for x in rows if x["mode"] == mode) / len(mcq)
        print(f"  {mode}: acc={acc:.3f}")

    per_item = pd.DataFrame(rows)
    wide = per_item.pivot(index="id", columns="mode", values="correct").astype(bool)
    summary = []
    for mode in args.modes:
        g = per_item[per_item["mode"] == mode]
        k = int(g["correct"].sum())
        lo, hi = wilson_ci(k, len(g))
        rec = {
            "mode": mode,
            "n": len(g),
            "correct": k,
            "accuracy": round(k / len(g), 4),
            "ci95_low": round(lo, 4),
            "ci95_high": round(hi, 4),
            "unparsed": int(g["pred"].isna().sum()),
            "errors": int((g["error"] != "").sum()),
        }
        if mode != "none" and "none" in wide:
            b, c, p = mcnemar_p(wide["none"], wide[mode])
            rec.update(
                {
                    "delta_vs_none": round((k - wide["none"].sum()) / len(g), 4),
                    "none_only_correct": b,
                    "mode_only_correct": c,
                    "mcnemar_p": round(p, 4),
                }
            )
        summary.append(rec)
    summary = pd.DataFrame(summary)

    suffix = "_smoke" if args.limit else ""
    out = results_dir(cfg)
    per_item.to_csv(out / f"mcq_per_item{suffix}.csv", index=False)
    summary.to_csv(out / f"mcq_summary{suffix}.csv", index=False)
    (out / f"mcq_meta{suffix}.json").write_text(
        json.dumps(
            {
                "dataset": ecfg["mcq_dataset"],
                "split": ecfg["mcq_split"],
                "n": len(mcq),
                "model": model,
                "top_k": top_k,
                "seed": cfg["seed"],
            },
            indent=2,
        )
    )
    print("\n" + summary.to_string(index=False))


if __name__ == "__main__":
    main()

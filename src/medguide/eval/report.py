"""Combine results/*.csv|json into results/ablation.md and one chart (results/ablation.png).

python -m medguide.eval.report
Only reads files produced by the eval scripts; sections whose inputs are missing are skipped.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from medguide.config import load_config

MODE_ORDER = ["none", "dense", "bm25", "hybrid", "hybrid_rerank"]


def _ordered(df: pd.DataFrame) -> pd.DataFrame:
    order = {m: i for i, m in enumerate(MODE_ORDER)}
    return df.sort_values("mode", key=lambda s: s.map(order)).reset_index(drop=True)


def _fmt(df: pd.DataFrame, pct_cols: list[str]) -> str:
    df = df.copy()
    for c in pct_cols:
        if c in df:
            df[c] = (df[c] * 100).map("{:.1f}".format)
    return df.to_markdown(index=False)


def retrieval_section(out: Path) -> tuple[str, pd.DataFrame | None]:
    path = out / "retrieval_summary.csv"
    if not path.exists():
        return "", None
    meta = json.loads((out / "retrieval_meta.json").read_text())
    df = _ordered(pd.read_csv(path))
    cols = [
        "mode",
        "n",
        "recall@1",
        "recall@3",
        "recall@5",
        "recall@10",
        "mrr@10",
        "doc_hit@5",
        "latency_ms_p50",
    ]
    text = [
        f"## Retrieval ({meta['label']}, n={meta['n_questions']})",
        "",
        "Recall@k / doc_hit@k in %, MRR@10 in %, latency in ms (single query, laptop).",
        "",
        _fmt(
            df[cols].round({"latency_ms_p50": 0}),
            ["recall@1", "recall@3", "recall@5", "recall@10", "mrr@10", "doc_hit@5"],
        ),
    ]
    src = out / "retrieval_by_source.csv"
    if src.exists():
        by = pd.read_csv(src).pivot(index="mode", columns="source", values="recall@5")
        by = _ordered(by.reset_index())
        text += [
            "",
            "Recall@5 by source (%):",
            "",
            _fmt(by, [c for c in by.columns if c != "mode"]),
        ]
    return "\n".join(text), df


def generation_section(out: Path) -> str:
    path = out / "generation_summary.json"
    if not path.exists():
        return ""
    s = json.loads(path.read_text())
    lines = [
        f"## Generation quality ({s['label']}, mode={s['mode']}, n={s['n_in_scope']})",
        "",
        f"- Generator: `{s['generator']}`; judge: `{s['judge']}` ({s['judge_reliability']})",
    ]
    if s.get("same_family_judge"):
        lines.append("- Judge and generator are the same model: scores may be biased upward.")
    for key, label in [
        ("citation_validity_rate", "Citation validity (all [n] resolve)"),
        ("answers_with_citation_rate", "Answers with at least one citation"),
        ("in_scope_abstention_rate", "In-scope questions where the model declined"),
        ("gold_chunk_in_context_rate", "Gold chunk among the 5 passages given to the LLM"),
    ]:
        if key in s:
            lines.append(f"- {label}: {s[key] * 100:.1f}%")
    if "ragas" in s:
        lines += ["", "| RAGAS metric | mean | NaN (judge failures) |", "|---|---|---|"]
        for k, v in s["ragas"].items():
            lines.append(f"| {k} | {v:.3f} | {s['ragas_nan_counts'].get(k, 0)} |")
    if "out_of_scope" in s:
        o = s["out_of_scope"]
        lines += [
            "",
            f"Out-of-scope questions (n={o['n']}): correctly declined "
            f"{o['correct_abstention_rate'] * 100:.0f}%",
        ]
        lines += [f"- {t}: {v * 100:.0f}%" for t, v in o["by_type"].items()]
    return "\n".join(lines)


def mcq_section(out: Path) -> tuple[str, pd.DataFrame | None]:
    path = out / "mcq_summary.csv"
    if not path.exists():
        return "", None
    meta = json.loads((out / "mcq_meta.json").read_text())
    df = _ordered(pd.read_csv(path))
    cols = [
        c
        for c in [
            "mode",
            "n",
            "accuracy",
            "ci95_low",
            "ci95_high",
            "delta_vs_none",
            "none_only_correct",
            "mode_only_correct",
            "mcnemar_p",
            "unparsed",
        ]
        if c in df
    ]
    text = "\n".join(
        [
            f"## MCQ: {meta['dataset']} ({meta['split']}, n={meta['n']}, model `{meta['model']}`)",
            "",
            "Accuracy and 95% Wilson CI in %. McNemar exact test vs `none` on the same items.",
            "",
            _fmt(df[cols], ["accuracy", "ci95_low", "ci95_high", "delta_vs_none"]),
        ]
    )
    return text, df


def make_chart(retrieval: pd.DataFrame | None, mcq: pd.DataFrame | None, path: Path) -> bool:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [p for p in (retrieval, mcq) if p is not None]
    if not panels:
        return False
    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 4), squeeze=False)
    ax_i = 0
    if retrieval is not None:
        ax = axes[0][ax_i]
        x = range(len(retrieval))
        width = 0.38
        ax.bar([i - width / 2 for i in x], retrieval["recall@5"] * 100, width, label="Recall@5")
        ax.bar([i + width / 2 for i in x], retrieval["mrr@10"] * 100, width, label="MRR@10")
        ax.set_xticks(list(x), retrieval["mode"], rotation=15)
        ax.set_ylabel("%")
        ax.set_ylim(0, 100)
        ax.set_title("Retrieval (synthetic questions)")
        ax.legend(frameon=False)
        ax_i += 1
    if mcq is not None:
        ax = axes[0][ax_i]
        acc = mcq["accuracy"] * 100
        err = [acc - mcq["ci95_low"] * 100, mcq["ci95_high"] * 100 - acc]
        ax.bar(mcq["mode"], acc, yerr=err, capsize=4, color="#5b8def")
        ax.set_ylabel("accuracy %")
        ax.set_ylim(0, 100)
        ax.set_title("MCQ accuracy (95% CI)")
        ax.tick_params(axis="x", rotation=15)
    for row in axes:
        for ax in row:
            ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def main() -> None:
    cfg = load_config()
    out = Path(cfg["paths"]["results_dir"])
    ret_text, ret_df = retrieval_section(out)
    gen_text = generation_section(out)
    mcq_text, mcq_df = mcq_section(out)
    parts = [
        "# GuideCite ablation results",
        "",
        "Generated by `make report` from the files in `results/`. Do not edit by hand.",
    ]
    parts += [t for t in (ret_text, gen_text, mcq_text) if t]
    if make_chart(ret_df, mcq_df, out / "ablation.png"):
        parts += ["", "![ablation](ablation.png)"]
    (out / "ablation.md").write_text("\n\n".join(parts) + "\n")
    print(f"wrote {out / 'ablation.md'}")


if __name__ == "__main__":
    main()

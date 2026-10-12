"""Run a few hand-written questions end to end and save the raw outputs (Phase 3 check).

Usage: python scripts/demo_answers.py [--mode hybrid_rerank]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from medguide.config import load_config
from medguide.generation.answer import AnswerEngine

QUESTIONS = [
    "Which groups of people should be screened for latent tuberculosis infection?",
    "Which vaccines are recommended for pregnant women?",
    "Should inhaled corticosteroids be used for chronic asthma in children under 12?",
    "How should laboratories test for Zika virus infection?",
    "What is the recommended treatment for a broken wrist in a cat?",  # should abstain
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default=None)
    args = parser.parse_args()
    cfg = load_config()
    engine = AnswerEngine(cfg)
    outputs = []
    for q in QUESTIONS:
        out = engine.answer(q, mode=args.mode)
        outputs.append({"question": q, **out})
        print("=" * 100 + f"\nQ: {q}\n\n{out['answer']}\n")
        for c in out["citations"]:
            print(f"  [{c['n']}] {c['source']} | {c['title'][:70]} | {(c['section'] or '-')[:40]}")
        print(
            f"  citations_valid={out['citations_valid']} invalid={out['invalid_citations']} "
            f"timings={out['timings']}"
        )
    path = Path(cfg["paths"]["results_dir"]) / "demo_answers.json"
    for o in outputs:  # keep the file small: drop full retrieved chunk texts
        o["retrieved"] = [
            {"chunk_id": r["chunk"]["chunk_id"], "title": r["chunk"]["title"], "score": r["score"]}
            for r in o["retrieved"]
        ]
    path.write_text(json.dumps(outputs, indent=2, ensure_ascii=False))
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()

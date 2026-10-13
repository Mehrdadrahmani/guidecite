"""Draw the GuideCite pipeline diagram.

Usage: python scripts/make_pipeline_diagram.py docs/pipeline.svg
       rsvg-convert -z 1.5 docs/pipeline.svg -o docs/pipeline.png
"""

import sys
from pathlib import Path
from xml.sax.saxutils import escape

W, H = 1400, 785
FONT = "-apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
out: list[str] = []


def add(s: str) -> None:
    out.append(s)


def box(x, y, w, h, title, lines=(), accent="#3B5BDB", fill="#FFFFFF", bold_lines=()):
    add(
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{fill}" '
        f'stroke="{accent}" stroke-width="1.6"/>'
    )
    n = 1 + len(lines)
    line_h = 17
    top = y + h / 2 - (n - 1) * line_h / 2 + 5
    cx = x + w / 2
    add(
        f'<text x="{cx}" y="{top}" text-anchor="middle" font-size="15" font-weight="700" '
        f'fill="#1F2933">{escape(title)}</text>'
    )
    for i, line in enumerate(lines, start=1):
        weight = "600" if line in bold_lines else "400"
        color = accent if line in bold_lines else "#52606D"
        add(
            f'<text x="{cx}" y="{top + i * line_h}" text-anchor="middle" font-size="12.5" '
            f'font-weight="{weight}" fill="{color}">{escape(line)}</text>'
        )


def arrow(x1, y1, x2, y2, dashed=False):
    dash = ' stroke-dasharray="6 5"' if dashed else ""
    add(
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#7B8794" stroke-width="1.8"'
        f'{dash} marker-end="url(#arrow)"/>'
    )


def lane(y, h, label, fill, stroke, color):
    add(
        f'<rect x="20" y="{y}" width="{W - 40}" height="{h}" rx="14" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="1.2"/>'
    )
    add(
        f'<text x="40" y="{y + 26}" font-size="12.5" font-weight="700" letter-spacing="1.4" '
        f'fill="{color}">{escape(label)}</text>'
    )


add(
    f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
    f'font-family="{FONT}">'
)
add(
    '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
    'markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" '
    'fill="#7B8794"/></marker></defs>'
)
add(f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>')
add(
    '<text x="40" y="44" font-size="22" font-weight="700" fill="#1F2933">GuideCite pipeline'
    '<tspan dx="14" font-weight="400" fill="#52606D">—</tspan><tspan dx="10" font-weight="400" fill="#52606D">answers from clinical guidelines, '
    "with citations</tspan></text>"
)

# Lane 1: offline indexing
BLUE = "#3B5BDB"
lane(70, 190, "1 · OFFLINE INDEXING", "#F4F7FF", "#D5DEF7", BLUE)
box(45, 112, 220, 124, "Guideline corpus", ["epfl-llm/guidelines", "WHO · CDC · NICE"], BLUE)
box(
    305,
    112,
    220,
    124,
    "Sample & clean",
    ["300 docs (100 per source)", "normalise text, infer titles"],
    BLUE,
)
box(
    565,
    112,
    230,
    124,
    "Structure-aware chunking",
    ["heading-aware, ~450 tokens", "64-token overlap", "8,634 chunks"],
    BLUE,
    bold_lines=("8,634 chunks",),
)
box(840, 108, 260, 62, "Dense index", ["bge-base-en-v1.5 → Qdrant"], BLUE)
box(840, 180, 260, 62, "Sparse index", ["BM25 Okapi"], BLUE)
arrow(265, 174, 303, 174)
arrow(525, 174, 563, 174)
arrow(795, 160, 838, 139)
arrow(795, 188, 838, 211)
add(
    f'<text x="1140" y="150" font-size="12.5" fill="#52606D">'
    f'<tspan x="1140" dy="0">Reproducible from scratch:</tspan>'
    f'<tspan x="1140" dy="20" font-family="Menlo, Consolas, monospace" fill="{BLUE}">make data</tspan>'
    f'<tspan x="1140" dy="20" font-family="Menlo, Consolas, monospace" fill="{BLUE}">make index</tspan>'
    f"</text>"
)

# Lane 2: online query
GREEN = "#0B7A55"
lane(300, 245, "2 · ONLINE QUERY  (FastAPI  POST /ask)", "#F1FAF6", "#CBE9DB", GREEN)
xs = [40, 208, 376, 544, 712, 880, 1048, 1216]
bw, by, bh = 140, 345, 170
cy = by + bh / 2
box(xs[0], by, bw, bh, "Question", ["free-text", "clinical query"], GREEN)
box(xs[1], by, bw, 78, "Dense search", ["top-30"], GREEN)
box(xs[1], by + 92, bw, 78, "BM25 search", ["top-30"], GREEN)
box(xs[2], by, bw, bh, "RRF fusion", ["reciprocal rank", "fusion, k = 60"], GREEN)
box(
    xs[3],
    by,
    bw,
    bh,
    "Rerank",
    ["cross-encoder", "bge-reranker-base", "25 → top 5"],
    GREEN,
    bold_lines=("25 → top 5",),
)
box(xs[4], by, bw, bh, "Grounded prompt", ["numbered passages", "cite [n] or abstain"], GREEN)
box(xs[5], by, bw, bh, "LLM", ["Qwen3-8B (local)", "via Ollama; any", "LangChain provider"], GREEN)
box(xs[6], by, bw, bh, "Citation check", ["every [n] must", "resolve to a", "passage"], GREEN)
box(xs[7], by, bw, bh, "Response", ["answer + sources", "+ disclaimer", "+ timings"], GREEN)
# connector: indexes are queried at runtime
add(
    '<path d="M970,242 L970,280 L334,280 L334,341" fill="none" stroke="#7B8794" '
    'stroke-width="1.6" stroke-dasharray="6 5" marker-end="url(#arrow)"/>'
)
add(
    '<text x="652" y="274" text-anchor="middle" font-size="12" font-style="italic" '
    'fill="#52606D">indexes queried at runtime</text>'
)

arrow(xs[0] + bw, cy - 12, xs[1] - 2, by + 39)
arrow(xs[0] + bw, cy + 12, xs[1] - 2, by + 131)
arrow(xs[1] + bw, by + 39, xs[2] - 2, cy - 12)
arrow(xs[1] + bw, by + 131, xs[2] - 2, cy + 12)
for i in range(2, 7):
    arrow(xs[i] + bw, cy, xs[i + 1] - 2, cy)
add(
    f'<text x="{W / 2}" y="535" text-anchor="middle" font-size="12" fill="#52606D">'
    "Retrieval mode is one config switch: "
    f'<tspan font-family="Menlo, Consolas, monospace" fill="{GREEN}">'
    "dense | bm25 | hybrid | hybrid_rerank</tspan></text>"
)

# Lane 3: evaluation
AMBER = "#B26A00"
lane(570, 195, "3 · EVALUATION  (outputs committed to results/)", "#FFF8EC", "#F2DDB4", AMBER)
ex, ew = [35, 375, 715, 1055], 310
eb, eh = 610, 135
box(
    ex[0],
    eb,
    ew,
    eh,
    "Retrieval ablation",
    ["201 synthetic questions", "4 retrieval modes", "Recall@k · MRR@10 · latency"],
    AMBER,
)
box(
    ex[1],
    eb,
    ew,
    eh,
    "Answer quality",
    [
        "50 questions, hybrid + rerank",
        "RAGAS: faithfulness, relevancy,",
        "context precision / recall",
        "citation validity",
    ],
    AMBER,
)
box(
    ex[2],
    eb,
    ew,
    eh,
    "Abstention",
    ["20 out-of-scope questions", "medical-not-in-corpus,", "non-medical, prompt injection"],
    AMBER,
)
box(
    ex[3],
    eb,
    ew,
    eh,
    "MedQA (USMLE)",
    ["300 questions", "with vs without retrieval", "Wilson 95% CI · McNemar test"],
    AMBER,
)

add("</svg>")
Path(sys.argv[1]).write_text("\n".join(out))

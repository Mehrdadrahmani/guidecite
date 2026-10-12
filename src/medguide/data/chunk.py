"""Structure-aware chunking: split on markdown headings, pack small sections,
recursively split oversized ones, and prefix each chunk with title + section."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass

import pandas as pd

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")

TokenCounter = Callable[[str], int]


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    chunk_index: int
    source: str
    title: str
    url: str | None
    section: str | None
    text: str  # prefixed text that is embedded and shown to the LLM
    n_tokens: int


@dataclass
class Section:
    heading: str | None
    body: str


def split_sections(text: str) -> list[Section]:
    """Split markdown text into (heading, body) sections; text before any heading has heading None."""
    sections: list[Section] = []
    heading: str | None = None
    lines: list[str] = []
    for line in text.split("\n"):
        match = _HEADING.match(line)
        if match:
            if "\n".join(lines).strip():
                sections.append(Section(heading, "\n".join(lines).strip()))
            heading, lines = match.group(2).strip(), []
        else:
            lines.append(line)
    if "\n".join(lines).strip():
        sections.append(Section(heading, "\n".join(lines).strip()))
    return sections


MAX_SECTION_WORDS = 15  # some "headings" in the dataset are whole sentences


def make_prefix(title: str, section: str | None) -> str:
    prefix = f"Title: {title}\n"
    if section:
        words = section.split()
        label = " ".join(words[:MAX_SECTION_WORDS]) + (
            "…" if len(words) > MAX_SECTION_WORDS else ""
        )
        prefix += f"Section: {label}\n"
    return prefix + "\n"


def _pack_sections(sections: list[Section], budget: int, count: TokenCounter) -> list[Section]:
    """Greedily merge consecutive small sections so chunks are not tiny fragments."""
    packed: list[Section] = []
    for sec in sections:
        block = f"{sec.heading}\n{sec.body}" if sec.heading else sec.body
        if packed and count(packed[-1].body) + count(block) <= budget:
            packed[-1] = Section(packed[-1].heading, packed[-1].body + "\n\n" + block)
        else:
            packed.append(Section(sec.heading, sec.body))
    return packed


def chunk_document(
    doc: dict,
    splitter,
    count: TokenCounter,
    chunk_tokens: int,
    min_chunk_tokens: int = 20,
) -> list[Chunk]:
    """Chunk one document. `splitter` must expose `split_text(str) -> list[str]`."""
    title = doc["title"]
    sections = split_sections(doc["text"]) or [Section(None, doc["text"])]
    budget = chunk_tokens - count(make_prefix(title, " ".join(["word"] * MAX_SECTION_WORDS)))
    chunks: list[Chunk] = []
    for sec in _pack_sections(sections, budget, count):
        pieces = [sec.body] if count(sec.body) <= budget else splitter.split_text(sec.body)
        for piece in pieces:
            piece = piece.strip()
            if count(piece) < min_chunk_tokens:
                continue
            text = make_prefix(title, sec.heading) + piece
            idx = len(chunks)
            chunks.append(
                Chunk(
                    chunk_id=f"{doc['doc_id']}:{idx:04d}",
                    doc_id=doc["doc_id"],
                    chunk_index=idx,
                    source=doc["source"],
                    title=title,
                    url=doc.get("url"),
                    section=sec.heading,
                    text=text,
                    n_tokens=count(text),
                )
            )
    return chunks


def build_token_counter(tokenizer_name: str) -> TokenCounter:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    return lambda s: len(tok.encode(s, add_special_tokens=False))


def build_splitter(tokenizer_name: str, chunk_tokens: int, overlap_tokens: int):
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    # leave room for the title/section prefix (up to ~80 tokens with OCR-noisy titles)
    return RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tok,
        chunk_size=chunk_tokens - 80,
        chunk_overlap=overlap_tokens,
        separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
    )


def chunk_corpus(docs: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    ccfg = cfg["chunking"]
    count = build_token_counter(ccfg["tokenizer"])
    splitter = build_splitter(ccfg["tokenizer"], ccfg["chunk_tokens"], ccfg["overlap_tokens"])
    rows: list[dict] = []
    for doc in docs.to_dict("records"):
        for ch in chunk_document(
            doc, splitter, count, ccfg["chunk_tokens"], ccfg["min_chunk_tokens"]
        ):
            rows.append(asdict(ch))
    return pd.DataFrame(rows)


def chunk_stats(chunks: pd.DataFrame) -> dict:
    lengths = chunks["n_tokens"]
    q = lengths.quantile([0.05, 0.25, 0.5, 0.75, 0.95]).round().astype(int).to_dict()
    return {
        "documents": int(chunks["doc_id"].nunique()),
        "chunks": int(len(chunks)),
        "chunks_per_source": chunks["source"].value_counts().to_dict(),
        "tokens_min": int(lengths.min()),
        "tokens_max": int(lengths.max()),
        "tokens_mean": round(float(lengths.mean()), 1),
        "tokens_quantiles": {f"p{int(k * 100)}": v for k, v in q.items()},
    }


def iter_records(chunks: pd.DataFrame) -> Iterable[dict]:
    yield from chunks.to_dict("records")

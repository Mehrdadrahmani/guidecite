"""Normalise, filter and deterministically sample guideline documents."""

from __future__ import annotations

import re

import pandas as pd

_MISSING = {"", "none", "nan", "null"}
_HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*#*\s*$", re.MULTILINE)


def _none_if_missing(value: object) -> str | None:
    """The dataset stores missing values as the literal string 'None'."""
    if value is None:
        return None
    text = str(value).strip()
    return None if text.lower() in _MISSING else text


# some headings are glued to the previous sentence: "...was reviewed.# BACKGROUND"
_GLUED_HEADING = re.compile(r"(?<=[^\s#])(#{1,6} )(?=[A-Z0-9])")


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\u00a0", " ")
    text = _GLUED_HEADING.sub(r"\n\1", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# headings that describe document structure rather than the guideline topic
_GENERIC_HEADING = re.compile(
    r"^(introduction|background|executive summary|summary|target audience|audience|contents|"
    r"table of contents|abbreviations|acronyms|acknowledg|foreword|preface|methods|scope|"
    r"overview|purpose|rationale|objectives|recommendations?\b|key recommendations|disclos|"
    r"references|table\b|box\b|figure\b|guideline development|contributors|risk group)",
    re.IGNORECASE,
)
_BOILERPLATE = re.compile(
    r"disclose|financial interest|continuing education|copyright|all rights reserved|"
    r"planners|commercial support|accredit|unlabeled use|investigational use",
    re.I,
)


def _clip(text: str, n: int = 110) -> str:
    text = " ".join(text.split()).strip(" .#")
    return text if len(text) <= n else text[:n].rsplit(" ", 1)[0] + "…"


def infer_title(text: str, source: str) -> str:
    """Best-effort title for docs without one (all CDC and WHO docs in this dataset).

    Use the opening heading if the document starts with a topical one; otherwise the first
    non-boilerplate sentence (mid-document headings are usually section names, not topics).
    """
    first = _HEADING.match(text)
    if first and len(first.group(1).split()) >= 3 and not _GENERIC_HEADING.match(first.group(1)):
        return _clip(first.group(1))
    body = _HEADING.sub("", text)
    for sentence in re.split(r"(?<=[.!?])\s+", body[:3000]):
        sentence = sentence.strip()
        if (
            len(sentence.split()) >= 6
            and sentence[:1].isupper()  # skip fragments cut mid-sentence
            and not _BOILERPLATE.search(sentence)
        ):
            return _clip(sentence)
    return f"{source.upper()} guideline"


def clean_documents(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["title", "url", "overview", "text"]:
        df[col] = df[col].map(_none_if_missing)
    df = df[df["text"].notna()].copy()
    df["text"] = df["text"].map(normalize_text)
    df["title_inferred"] = [not (isinstance(t, str) and t) for t in df["title"]]
    df["title"] = [
        infer_title(text, src) if inferred else t
        for t, text, src, inferred in zip(
            df["title"], df["text"], df["source"], df["title_inferred"], strict=True
        )
    ]
    df["n_words"] = df["text"].str.split().str.len()
    return df.drop_duplicates(subset="text").reset_index(drop=True)


def filter_and_sample(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Drop too-short/too-long docs, then sample a balanced, seeded subset per source."""
    dcfg = cfg["data"]
    df = df[(df["n_words"] >= dcfg["min_words"]) & (df["n_words"] <= dcfg["max_words"])]
    parts = []
    for source in dcfg["sources"]:
        sub = df[df["source"] == source].sort_values("id")
        n = min(dcfg["docs_per_source"], len(sub))
        parts.append(sub.sample(n=n, random_state=cfg["seed"]))
    out = pd.concat(parts).sort_values(["source", "id"]).reset_index(drop=True)
    return out.rename(columns={"id": "doc_id"})

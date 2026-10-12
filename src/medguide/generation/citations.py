"""Parse [n] citation markers and validate them against the retrieved passages."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

# matches [1], [2, 3], [1-3], [1][2]
_CITE = re.compile(r"\[(\d+(?:\s*[-,–]\s*\d+)*)\]")


@dataclass
class Citation:
    n: int
    title: str
    source: str
    url: str | None
    section: str | None
    snippet: str
    chunk_id: str


@dataclass
class CitationReport:
    citations: list[Citation]
    invalid: list[int]  # cited numbers that do not map to a retrieved passage

    @property
    def all_valid(self) -> bool:
        return not self.invalid

    def to_dicts(self) -> list[dict]:
        return [asdict(c) for c in self.citations]


def parse_citation_numbers(text: str) -> list[int]:
    """Return cited numbers in order of first appearance (ranges expanded)."""
    seen: list[int] = []
    for group in _CITE.findall(text):
        for part in re.split(r"\s*,\s*", group):
            bounds = re.split(r"\s*[-–]\s*", part)
            if len(bounds) == 2:
                lo, hi = int(bounds[0]), int(bounds[1])
                nums = range(lo, hi + 1) if 0 < hi - lo < 20 else [lo, hi]
            else:
                nums = [int(bounds[0])]
            for n in nums:
                if n not in seen:
                    seen.append(n)
    return seen


def _snippet(text: str, n_chars: int = 240) -> str:
    # drop the "Title: ... / Section: ..." prefix that every chunk carries
    body = text.split("\n\n", 1)[-1] if text.startswith("Title:") else text
    body = " ".join(body.split())
    return body if len(body) <= n_chars else body[:n_chars].rsplit(" ", 1)[0] + "…"


def resolve_citations(answer: str, passages: list[dict]) -> CitationReport:
    """Map [n] (1-based) to passages; numbers outside 1..len(passages) are reported invalid."""
    citations: list[Citation] = []
    invalid: list[int] = []
    for n in parse_citation_numbers(answer):
        if 1 <= n <= len(passages):
            p = passages[n - 1]
            citations.append(
                Citation(
                    n=n,
                    title=p["title"],
                    source=p["source"].upper(),
                    url=p.get("url"),
                    section=p.get("section"),
                    snippet=_snippet(p["text"]),
                    chunk_id=p["chunk_id"],
                )
            )
        else:
            invalid.append(n)
    return CitationReport(sorted(citations, key=lambda c: c.n), invalid)

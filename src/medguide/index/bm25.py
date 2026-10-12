"""BM25 index over chunks. Tokenisation keeps numbers, doses and drug names intact."""

from __future__ import annotations

import pickle
import re
from pathlib import Path

# word chars joined by - . / so "covid-19", "2.5", "mg/kg", "hba1c" stay single tokens
_TOKEN = re.compile(r"[a-z0-9]+(?:[-./][a-z0-9]+)*")

# small English stopword list; per-language lists can be added for Persian later
_EN_STOPWORDS = (
    "a an and are as at be by for from has have in is it its of on or that the this to was "
    "were will with which what who whom should can do does how when where why"
)
STOPWORDS = {"en": frozenset(_EN_STOPWORDS.split())}


def tokenize(text: str, language: str = "en", lowercase: bool = True) -> list[str]:
    if lowercase:
        text = text.lower()
    stop = STOPWORDS.get(language, set())
    return [t for t in _TOKEN.findall(text) if t not in stop]


class BM25Index:
    def __init__(self, records: list[dict], language: str = "en") -> None:
        from rank_bm25 import BM25Okapi

        self.language = language
        self.records = records
        self.bm25 = BM25Okapi([tokenize(r["text"], language) for r in records])

    def search(self, query: str, k: int) -> list[tuple[dict, float]]:
        import numpy as np

        scores = self.bm25.get_scores(tokenize(query, self.language))
        top = np.argsort(-scores)[:k]
        return [(self.records[i], float(scores[i])) for i in top if scores[i] > 0]

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str | Path) -> BM25Index:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"No BM25 index at {path}. Run `make index` first.")
        with open(path, "rb") as f:
            return pickle.load(f)

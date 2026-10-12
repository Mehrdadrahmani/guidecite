import pandas as pd
import pytest

from medguide.eval.mcq_eval import mcnemar_p, parse_letter, wilson_ci
from medguide.eval.retrieval_eval import item_metrics


@pytest.mark.parametrize(
    "text,expected",
    [
        ("reasoning...\nAnswer: C", "C"),
        ("The answer is **B**.", "B"),
        ("Answer: (D)", "D"),
        ("Answer: A\n...wait\nAnswer: B", "B"),  # last one wins
        ("A", "A"),
        ("I think A or maybe B", None),
        ("", None),
    ],
)
def test_parse_letter(text, expected):
    assert parse_letter(text) == expected


def test_wilson_ci_brackets_point_estimate():
    lo, hi = wilson_ci(150, 300)
    assert lo < 0.5 < hi and hi - lo < 0.13


def test_mcnemar_counts_discordant_pairs():
    base = pd.Series([True, True, False, False, True])
    other = pd.Series([True, False, True, True, True])
    b, c, p = mcnemar_p(base, other)
    assert (b, c) == (1, 2) and 0 < p <= 1


def test_retrieval_item_metrics():
    m = item_metrics("c3", "d2", ["c1", "c2", "c3"], ["d1", "d2"])
    assert m["rank"] == 3 and m["rr@10"] == pytest.approx(1 / 3)
    assert m["recall@1"] == 0 and m["recall@3"] == 1 and m["recall@5"] == 1
    assert m["doc_hit@1"] == 0 and m["doc_hit@3"] == 1
    miss = item_metrics("cx", "dx", ["c1"], ["d1"])
    assert miss["rank"] is None and miss["rr@10"] == 0 and miss["recall@10"] == 0

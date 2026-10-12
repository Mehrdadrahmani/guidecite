import pytest

from medguide.retrieval.hybrid import rrf_fuse


def test_rrf_scores_match_formula():
    fused = dict(rrf_fuse([["a", "b", "c"], ["b", "a", "d"]], k=60))
    assert fused["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert fused["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused["c"] == pytest.approx(1 / 63)
    assert fused["d"] == pytest.approx(1 / 63)


def test_rrf_rewards_agreement():
    order = [d for d, _ in rrf_fuse([["x", "a", "b"], ["y", "a", "b"]], k=60)]
    # "a" is 2nd in both lists and beats items that are 1st in only one list
    assert order[0] == "a"


def test_rrf_handles_empty_lists():
    assert rrf_fuse([[], []]) == []
    assert [d for d, _ in rrf_fuse([["a"], []])] == ["a"]

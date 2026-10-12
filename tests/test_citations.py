from medguide.generation.citations import parse_citation_numbers, resolve_citations

PASSAGES = [
    {
        "chunk_id": f"d:{i}",
        "title": f"T{i}",
        "source": "who",
        "url": None,
        "section": None,
        "text": f"Title: T{i}\n\nbody {i}",
    }
    for i in range(3)
]


def test_parse_formats():
    assert parse_citation_numbers("a [1] b [2][3] c [1, 3] d [2-3]") == [1, 2, 3]
    assert parse_citation_numbers("no citations here") == []


def test_valid_citations_resolve():
    rep = resolve_citations("Do X [1]. Avoid Y [3].", PASSAGES)
    assert rep.all_valid
    assert [c.n for c in rep.citations] == [1, 3]
    assert rep.citations[0].chunk_id == "d:0"
    assert rep.citations[1].snippet == "body 2"
    assert rep.citations[0].source == "WHO"


def test_invalid_ids_rejected():
    rep = resolve_citations("Claim [2]. Made up [4]. Zero [0].", PASSAGES)
    assert not rep.all_valid
    assert sorted(rep.invalid) == [0, 4]
    assert [c.n for c in rep.citations] == [2]


def test_strip_think_blocks():
    from medguide.generation.llm import strip_think

    assert strip_think("</think>\n\nOK") == "OK"
    assert strip_think("<think>\nreasoning [9]\n</think>\nAnswer [1].") == "Answer [1]."
    assert strip_think("plain answer") == "plain answer"

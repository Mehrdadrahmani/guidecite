from medguide.data.chunk import chunk_document, split_sections


def count(s: str) -> int:
    return len(s.split())


class WordSplitter:
    def __init__(self, size: int, overlap: int) -> None:
        self.size, self.overlap = size, overlap

    def split_text(self, text: str) -> list[str]:
        words = text.split()
        step = self.size - self.overlap
        return [" ".join(words[i : i + self.size]) for i in range(0, len(words), step)]


DOC = {
    "doc_id": "abc",
    "source": "nice",
    "title": "Hypertension in adults",
    "url": "https://example.org/ng136",
    "text": (
        "Intro paragraph with some words here.\n\n"
        "# Diagnosis\n" + "measure blood pressure " * 10 + "\n\n"
        "## Treatment\n" + "offer an ACE inhibitor 5 mg " * 200 + "\n\n"
        "# Empty heading\n\n"
        "# References\n" + "ref " * 30
    ),
}


def make_chunks(chunk_tokens: int = 120):
    return chunk_document(DOC, WordSplitter(80, 10), count, chunk_tokens, min_chunk_tokens=3)


def test_split_sections_tracks_headings():
    secs = split_sections(DOC["text"])
    assert secs[0].heading is None
    assert [s.heading for s in secs[1:]] == ["Diagnosis", "Treatment", "References"]


def test_no_empty_chunks_and_metadata_present():
    chunks = make_chunks()
    assert len(chunks) > 3
    for i, ch in enumerate(chunks):
        assert ch.text.strip()
        assert ch.chunk_index == i
        assert ch.chunk_id == f"abc:{i:04d}"
        assert ch.doc_id == "abc" and ch.source == "nice" and ch.url
        assert ch.text.startswith("Title: Hypertension in adults\n")


def test_long_section_is_split_and_keeps_section():
    chunks = make_chunks()
    treatment = [c for c in chunks if c.section == "Treatment"]
    assert len(treatment) > 1
    assert all("Section: Treatment" in c.text for c in treatment)


def test_short_doc_is_single_chunk():
    doc = dict(DOC, text="Just one short paragraph about fever management in children.")
    chunks = chunk_document(doc, WordSplitter(80, 10), count, 120, min_chunk_tokens=3)
    assert len(chunks) == 1 and chunks[0].section is None


def test_glued_heading_is_split_and_missing_title_inferred():
    import pandas as pd

    from medguide.data.clean import clean_documents

    raw = pd.DataFrame(
        [
            {
                "id": "x",
                "source": "cdc",
                "title": "None",
                "url": "None",
                "overview": "None",
                "text": "# Acrylamide exposure criteria\nIntro was reviewed.# BACKGROUND\nBody.",
            }
        ]
    )
    doc = clean_documents(raw).iloc[0]
    assert doc["title"] == "Acrylamide exposure criteria"
    assert doc["url"] is None
    assert [s.heading for s in split_sections(doc["text"])] == [
        "Acrylamide exposure criteria",
        "BACKGROUND",
    ]

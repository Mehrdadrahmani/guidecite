"""API tests with a fake retriever and a fake LLM: no models, no network."""

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from medguide.api.main import create_app
from medguide.generation.answer import AnswerEngine
from medguide.retrieval.pipeline import Hit

CHUNKS = [
    {
        "chunk_id": "doc1:0000",
        "doc_id": "doc1",
        "chunk_index": 0,
        "source": "nice",
        "title": "Hypertension in adults",
        "url": "https://example.org/ng136",
        "section": "Treatment",
        "text": "Title: Hypertension in adults\nSection: Treatment\n\nOffer an ACE inhibitor.",
    },
    {
        "chunk_id": "doc2:0003",
        "doc_id": "doc2",
        "chunk_index": 3,
        "source": "who",
        "title": "WHO hypertension guideline",
        "url": None,
        "section": None,
        "text": "Title: WHO hypertension guideline\n\nStart drug treatment at 140/90 mmHg.",
    },
]


class FakeRetriever:
    def warmup(self, mode):
        pass

    def retrieve(self, query, mode=None, top_k=None):
        return [Hit(c, 1.0 - i * 0.1) for i, c in enumerate(CHUNKS[: top_k or 5])]


class FakeLLM:
    def __init__(self, reply: str) -> None:
        self.reply = reply

    def invoke(self, messages):
        assert "<context>" in messages[1][1]
        return SimpleNamespace(content=self.reply)


def make_client(reply: str) -> TestClient:
    factory = lambda cfg: AnswerEngine(cfg, retriever=FakeRetriever(), llm=FakeLLM(reply))  # noqa: E731
    return TestClient(create_app(engine_factory=factory))


def test_health():
    with make_client("x") as client:
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert body["model"] and body["default_mode"]
        assert "not medical advice" in body["disclaimer"]


def test_web_ui_is_served():
    with make_client("x") as client:
        res = client.get("/")
        assert res.status_code == 200
        assert res.headers["content-type"].startswith("text/html")
        assert "GuideCite" in res.text and "Not medical advice" in res.text


def test_ask_returns_structured_answer():
    reply = "NICE [1] recommends an ACE inhibitor; WHO [2] sets 140/90 mmHg."
    with make_client(reply) as client:
        r = client.post("/ask", json={"question": "First-line hypertension drug?", "mode": "bm25"})
        assert r.status_code == 200
        body = r.json()
        assert body["mode"] == "bm25"
        assert body["citations_valid"] is True
        assert [c["n"] for c in body["citations"]] == [1, 2]
        assert body["citations"][0]["source"] == "NICE"
        assert len(body["retrieved"]) == 2
        assert set(body["timings"]) == {"retrieval_ms", "generation_ms", "total_ms"}


def test_ask_flags_invalid_citation():
    with make_client("Claim [7].") as client:
        body = client.post("/ask", json={"question": "anything?"}).json()
        assert body["citations_valid"] is False and body["invalid_citations"] == [7]


@pytest.mark.parametrize(
    "payload",
    [
        {"question": ""},
        {"question": "x" * 5000},
        {"question": "ok?", "mode": "magic"},
        {"question": "ok?", "top_k": 0},
    ],
)
def test_input_validation(payload):
    with make_client("x") as client:
        assert client.post("/ask", json=payload).status_code == 422

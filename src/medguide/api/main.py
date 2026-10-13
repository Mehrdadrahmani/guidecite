"""GuideCite HTTP API: web UI at /, POST /ask, GET /health."""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from medguide.config import ConfigError, get_config
from medguide.generation.answer import DISCLAIMER, AnswerEngine

log = logging.getLogger("guidecite.api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

_CFG = get_config()
_INDEX_HTML = Path(__file__).parent / "static" / "index.html"
Mode = Literal["dense", "bm25", "hybrid", "hybrid_rerank"]


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=_CFG["api"]["max_question_chars"])
    mode: Mode | None = None
    top_k: int | None = Field(default=None, ge=1, le=_CFG["api"]["max_top_k"])


class CitationOut(BaseModel):
    n: int
    title: str
    source: str
    url: str | None = None
    section: str | None = None
    snippet: str
    chunk_id: str


class RetrievedOut(BaseModel):
    chunk: dict
    score: float


class AskResponse(BaseModel):
    answer: str
    citations: list[CitationOut]
    invalid_citations: list[int]
    citations_valid: bool
    retrieved: list[RetrievedOut]
    timings: dict[str, float]
    mode: str
    model: str
    disclaimer: str = DISCLAIMER


def create_app(engine_factory=None) -> FastAPI:
    """`engine_factory(cfg) -> AnswerEngine`; tests inject a mocked LLM through it."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        cfg = get_config()
        factory = engine_factory or (lambda c: AnswerEngine(c))
        try:
            engine = factory(cfg)
            engine.retriever.warmup(cfg["retrieval"]["mode"])  # load models once
            app.state.engine, app.state.error = engine, None
            log.info(
                "engine ready: mode=%s model=%s", cfg["retrieval"]["mode"], cfg["llm"]["model"]
            )
        except (ConfigError, FileNotFoundError) as exc:
            # start anyway so /health can report what is missing
            app.state.engine, app.state.error = None, str(exc)
            log.error("engine not ready: %s", exc)
        yield

    app = FastAPI(
        title="GuideCite",
        description="Answers from clinical guidelines, with citations. " + DISCLAIMER,
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        rid = uuid.uuid4().hex[:8]
        t0 = time.perf_counter()
        response = await call_next(request)
        ms = (time.perf_counter() - t0) * 1000
        log.info(
            "rid=%s %s %s -> %s %.0fms",
            rid,
            request.method,
            request.url.path,
            response.status_code,
            ms,
        )
        response.headers["X-Request-ID"] = rid
        return response

    @app.get("/", include_in_schema=False)
    def web_ui() -> FileResponse:
        return FileResponse(_INDEX_HTML, media_type="text/html")

    @app.get("/health")
    def health(request: Request) -> dict:
        state, cfg = request.app.state, get_config()
        return {
            "status": "ok" if state.engine else "degraded",
            "error": state.error,
            "model": f"{cfg['llm']['provider']}:{cfg['llm']['model']}",
            "default_mode": cfg["retrieval"]["mode"],
            "disclaimer": DISCLAIMER,
        }

    @app.post("/ask", response_model=AskResponse)
    async def ask(req: AskRequest, request: Request) -> dict:
        engine = request.app.state.engine
        if engine is None:
            raise HTTPException(503, detail=f"Engine not ready: {request.app.state.error}")
        try:
            return await run_in_threadpool(engine.answer, req.question, req.mode, req.top_k)
        except ConfigError as exc:
            raise HTTPException(503, detail=str(exc)) from exc

    return app


app = create_app()

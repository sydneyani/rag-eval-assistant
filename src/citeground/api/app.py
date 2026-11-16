"""FastAPI application.

`create_app` is the test seam. The module-level `app` is what uvicorn loads.
A missing index does not crash startup: `/v1/health` reports it, and
`/v1/answer` returns 503 until `citeground-ingest` has been run.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, field_validator

from citeground.config import Settings, get_settings
from citeground.errors import IndexNotFoundError
from citeground.generation.answer import AnswerService
from citeground.index.store import CorpusIndex
from citeground.llm.factory import build_chat_model, build_embedder, resolve_provider_name
from citeground.observability.tracing import MetricsRegistry, TraceSink
from citeground.retrieval.service import HybridRetriever
from citeground.schemas import AnswerResult

logger = logging.getLogger("citeground.api")
_STATIC = Path(__file__).resolve().parent / "static" / "index.html"


class AnswerRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    mode: str | None = None
    rerank: bool | None = None

    @field_validator("question")
    @classmethod
    def question_not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("question must not be blank")
        return stripped

    @field_validator("mode")
    @classmethod
    def known_mode(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if value not in {"hybrid", "bm25", "dense"}:
            raise ValueError("mode must be hybrid, bm25, or dense")
        return value


class HealthResponse(BaseModel):
    status: str
    index_loaded: bool
    chunks: int
    llm_provider: str
    embedding_provider: str
    retrieval_mode: str
    model: str


@dataclass
class Runtime:
    settings: Settings
    service: AnswerService | None
    metrics: MetricsRegistry
    index_chunks: int
    llm_provider: str
    embedding_provider: str
    model: str
    retrieval_mode: str


def build_runtime(
    settings: Settings,
    *,
    index: CorpusIndex | None = None,
    retriever: HybridRetriever | None = None,
    chat=None,
    load_index: bool = True,
) -> Runtime:
    metrics = MetricsRegistry()
    traces = TraceSink(settings.resolve(settings.traces_path))
    loaded = index
    if loaded is None and retriever is None and load_index:
        try:
            loaded = CorpusIndex.load(settings.resolve(settings.index_dir))
        except IndexNotFoundError:
            loaded = None
    if retriever is None and loaded is not None:
        embedder = build_embedder(settings)
        retriever = HybridRetriever(
            loaded,
            embedder,
            mode=settings.retrieval_mode,
            rerank_results=settings.rerank,
            rrf_k=settings.rrf_k,
            candidate_k=settings.candidate_k,
            vector_backend=settings.vector_store,
        )
    if retriever is None:
        offline_chat = chat or build_chat_model(settings)
        embedding_name = resolve_provider_name(
            settings.embedding_provider,
            api_key=settings.openai_api_key,
            offline_name="hashing",
        )
        return Runtime(
            settings=settings,
            service=None,
            metrics=metrics,
            index_chunks=0,
            llm_provider=offline_chat.provider_name,
            embedding_provider=embedding_name,
            model=offline_chat.model_name,
            retrieval_mode=settings.retrieval_mode,
        )
    chat_model = chat or build_chat_model(settings)
    service = AnswerService(retriever, chat_model, settings, traces)
    return Runtime(
        settings=settings,
        service=service,
        metrics=metrics,
        index_chunks=len(retriever.index.chunks),
        llm_provider=chat_model.provider_name,
        embedding_provider=retriever.embedder.provider_name,
        model=chat_model.model_name,
        retrieval_mode=retriever.mode,
    )


def create_app(runtime: Runtime | None = None, settings: Settings | None = None) -> FastAPI:
    resolved_settings = settings or get_settings()
    holder = runtime or build_runtime(resolved_settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.runtime = holder
        logger.info(
            "startup chunks=%s llm=%s embedding=%s",
            holder.index_chunks,
            holder.llm_provider,
            holder.embedding_provider,
        )
        yield

    app = FastAPI(
        title="CiteGround",
        version="0.1.0",
        summary="Citation-grounded answers over the Helios filing corpus.",
        lifespan=lifespan,
    )
    app.state.runtime = holder
    app.add_middleware(
        CORSMiddleware,
        allow_origins=holder.settings.cors_origin_list,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def trace_requests(request: Request, call_next):
        trace_id = request.headers.get("x-trace-id") or uuid4().hex
        request.state.trace_id = trace_id
        started = perf_counter()
        response = await call_next(request)
        latency_ms = (perf_counter() - started) * 1000
        response.headers["x-trace-id"] = trace_id
        holder.metrics.record_http(response.status_code, latency_ms)
        logger.info(
            "request",
            extra={
                "trace_id": trace_id,
                "path": request.url.path,
                "status": response.status_code,
                "latency_ms": round(latency_ms, 2),
            },
        )
        return response

    @app.get("/", response_class=HTMLResponse)
    def home() -> str:
        return _STATIC.read_text(encoding="utf-8")

    @app.get("/v1/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return _health(holder)

    @app.get("/health", response_model=HealthResponse, include_in_schema=False)
    def health_alias() -> HealthResponse:
        return _health(holder)

    @app.get("/v1/metrics")
    def metrics() -> dict[str, float | int | str]:
        snapshot = holder.metrics.snapshot()
        snapshot["llm_provider"] = holder.llm_provider
        snapshot["embedding_provider"] = holder.embedding_provider
        return snapshot

    @app.post("/v1/answer", response_model=AnswerResult)
    def answer(body: AnswerRequest, request: Request) -> AnswerResult:
        if holder.service is None:
            raise HTTPException(
                status_code=503,
                detail="Index is not loaded. Run `python -m citeground.cli.ingest` and restart.",
            )
        result = holder.service.answer(
            body.question,
            top_k=body.top_k,
            mode=body.mode,
            rerank=body.rerank,
            trace_id=getattr(request.state, "trace_id", None),
        )
        holder.metrics.record_answer(
            latency_ms=result.latency_ms,
            usage=result.usage,
            abstained=result.abstained,
        )
        return result

    return app


def _health(runtime: Runtime) -> HealthResponse:
    return HealthResponse(
        status="ok",
        index_loaded=runtime.service is not None,
        chunks=runtime.index_chunks,
        llm_provider=runtime.llm_provider,
        embedding_provider=runtime.embedding_provider,
        retrieval_mode=runtime.retrieval_mode,
        model=runtime.model,
    )


app = create_app()

import json
from pathlib import Path

from fastapi.testclient import TestClient

from citeground.api.app import build_runtime, create_app
from citeground.config import Settings
from citeground.generation.answer import parse_citations
from citeground.ingest.pipeline import ingest_corpus
from citeground.observability.cost import estimate_cost
from citeground.schemas import Chunk, ScoredChunk

CORPUS = Path(__file__).resolve().parents[1] / "data" / "corpus"
REVENUE = "What was fiscal 2024 revenue compared with fiscal 2023 revenue?"


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        llm_provider="mock",
        embedding_provider="hashing",
        embedding_dim=128,
        index_dir=tmp_path / "index",
        traces_path=tmp_path / "traces" / "traces.jsonl",
        chunker="recursive",
        chunk_size=900,
        chunk_overlap=120,
    )


def _client(tmp_path: Path, *, load_index: bool) -> TestClient:
    settings = _settings(tmp_path)
    if load_index:
        ingest_corpus(settings, corpus_dir=CORPUS, index_dir=settings.index_dir)
    runtime = build_runtime(settings, load_index=load_index)
    return TestClient(create_app(runtime))


def test_health_and_answer_without_an_index(tmp_path: Path) -> None:
    with _client(tmp_path, load_index=False) as client:
        health = client.get("/v1/health")
        assert health.status_code == 200
        assert health.json()["index_loaded"] is False
        assert health.json()["llm_provider"] == "mock"
        denied = client.post("/v1/answer", json={"question": REVENUE})
        assert denied.status_code == 503


def test_answer_cites_the_results_section_and_records_a_trace(tmp_path: Path) -> None:
    with _client(tmp_path, load_index=True) as client:
        page = client.get("/")
        assert page.status_code == 200
        assert "CiteGround" in page.text
        assert "Synthetic corpus" in page.text

        response = client.post(
            "/v1/answer", json={"question": REVENUE, "top_k": 5, "mode": "hybrid"}
        )
        assert response.status_code == 200
        assert response.headers["x-trace-id"]
        body = response.json()
        assert "4.82 billion" in body["answer"]
        assert body["abstained"] is False
        assert body["usage"]["cost_usd"] == 0
        assert body["usage"]["cost_basis"] == "offline-zero"
        assert body["usage"]["prompt_tokens"] > 0
        sections = {item["section_id"] for item in body["citations"]}
        assert "helios-10k-fy2024/results" in sections
        assert [stage["name"] for stage in body["stages"]] == ["retrieve", "generate"]

        abstained = client.post(
            "/v1/answer",
            json={"question": "What is the chief executive officer's favorite color?"},
        )
        assert abstained.status_code == 200
        assert abstained.json()["abstained"] is True

        metrics = client.get("/v1/metrics").json()
        assert metrics["answers"] == 2
        assert metrics["abstentions"] == 1
        assert metrics["cost_usd"] == 0

    traces = (tmp_path / "traces" / "traces.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(traces) == 2
    record = json.loads(traces[0])
    assert record["trace_id"] == body["trace_id"]
    assert "helios-10k-fy2024/results" in record["retrieved_section_ids"]


def test_blank_question_is_rejected(tmp_path: Path) -> None:
    with _client(tmp_path, load_index=True) as client:
        response = client.post("/v1/answer", json={"question": "   "})
        assert response.status_code == 422


def test_citation_parser_ignores_markers_outside_the_retrieved_set() -> None:
    hit = ScoredChunk(
        chunk=Chunk(
            chunk_id="doc/s#000",
            doc_id="doc",
            section_id="doc/s",
            doc_title="Doc",
            section_title="Section",
            text="Fiscal 2024 revenue was $4.82 billion.",
            position=0,
        ),
        score=0.5,
    )
    citations = parse_citations("Revenue was $4.82 billion. [1] Also [2] and [1].", [hit])
    assert [item.marker for item in citations] == [1]
    assert citations[0].section_id == "doc/s"


def test_cost_is_zero_offline_and_priced_for_openai() -> None:
    settings = Settings(_env_file=None)
    offline, basis = estimate_cost(
        settings,
        prompt_tokens=1000,
        completion_tokens=1000,
        embedding_tokens=1000,
        chat_provider="mock",
        embedding_provider="hashing",
    )
    assert offline == 0
    assert basis == "offline-zero"
    priced, priced_basis = estimate_cost(
        settings,
        prompt_tokens=1_000_000,
        completion_tokens=1_000_000,
        embedding_tokens=1_000_000,
        chat_provider="openai",
        embedding_provider="openai",
    )
    assert priced == settings.price_chat_input_per_mtok + settings.price_chat_output_per_mtok + (
        settings.price_embedding_per_mtok
    )
    assert priced_basis == "list-price-estimate"

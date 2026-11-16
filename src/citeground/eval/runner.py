"""Run the labeled set through the same answer path the API uses.

Each experiment gets a fresh index. Retrieval metrics use the top 10 chunks.
The answer is generated from the top 5, which is the product default, so the
answer metrics describe the system a user actually calls.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from citeground.config import Settings
from citeground.eval.dataset import LabeledQuestion, load_dataset
from citeground.eval.experiments import Aggregate, ExperimentSpec
from citeground.eval.judge import DeterministicJudge, Judge
from citeground.eval.metrics import mean_reciprocal_rank, recall_at_k
from citeground.generation.answer import AnswerService, is_abstention
from citeground.index.store import CorpusIndex
from citeground.ingest.pipeline import ingest_corpus
from citeground.llm.factory import build_chat_model, build_embedder
from citeground.observability.tracing import TraceSink
from citeground.retrieval.service import HybridRetriever


def run_evaluation(
    *,
    settings: Settings,
    dataset_path: Path,
    specs: list[ExperimentSpec],
    work_dir: Path,
    judge: Judge | None = None,
    corpus_dir: Path | None = None,
) -> dict:
    questions = load_dataset(dataset_path)
    chosen = judge or DeterministicJudge()
    experiments = []
    for spec in specs:
        experiments.append(
            _run_spec(
                settings=settings,
                questions=questions,
                spec=spec,
                work_dir=work_dir / spec.name,
                judge=chosen,
                corpus_dir=corpus_dir,
            )
        )
    answerable = sum(1 for question in questions if not question.expect_abstention)
    chat = build_chat_model(settings)
    embedder = build_embedder(settings)
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "judge": chosen.name,
        "embedding_provider": embedder.provider_name,
        "embedding_model": embedder.model_name,
        "llm_provider": chat.provider_name,
        "llm_model": chat.model_name,
        "dataset": str(dataset_path),
        "n_questions": len(questions),
        "n_answerable": answerable,
        "n_unanswerable": len(questions) - answerable,
        "label": (
            "Offline run. No API key. Faithfulness and answer relevance are "
            "deterministic lexical proxies, not RAGAS model scores."
            if chosen.name == "offline-deterministic"
            else "LLM-judge run. Faithfulness and answer relevance in the extra fields are model scores."
        ),
        "experiments": experiments,
    }


def _run_spec(
    *,
    settings: Settings,
    questions: list[LabeledQuestion],
    spec: ExperimentSpec,
    work_dir: Path,
    judge: Judge,
    corpus_dir: Path | None,
) -> dict:
    index_dir = work_dir / "index"
    spec_settings = settings.model_copy(
        update={
            "chunker": spec.chunker,
            "chunk_size": spec.chunk_size,
            "chunk_overlap": spec.chunk_overlap,
            "retrieval_mode": spec.retrieval_mode,
            "rerank": spec.rerank,
            "index_dir": index_dir,
            "traces_path": work_dir / "traces.jsonl",
            "top_k": spec.answer_k,
        }
    )
    ingest_corpus(spec_settings, corpus_dir=corpus_dir, index_dir=index_dir)
    index = CorpusIndex.load(index_dir)
    embedder = build_embedder(spec_settings)
    retriever = HybridRetriever(
        index,
        embedder,
        mode=spec.retrieval_mode,
        rerank_results=spec.rerank,
        rrf_k=spec_settings.rrf_k,
        candidate_k=max(spec_settings.candidate_k, spec.retrieve_k),
        vector_backend=spec_settings.vector_store,
    )
    chat = build_chat_model(spec_settings)
    service = AnswerService(
        retriever,
        chat,
        spec_settings,
        TraceSink(spec_settings.traces_path),
    )
    rows: list[dict] = []
    for question in questions:
        hits = retriever.retrieve(
            question.question,
            spec.retrieve_k,
            mode=spec.retrieval_mode,
            rerank_results=spec.rerank,
        )
        sections = [hit.chunk.section_id for hit in hits]
        result = service.answer(
            question.question,
            top_k=spec.answer_k,
            mode=spec.retrieval_mode,
            rerank=spec.rerank,
        )
        contexts = [hit.chunk.text for hit in result.retrieved]
        judged = judge.score(
            question=question.question,
            answer=result.answer,
            contexts=contexts,
            gold_answer=question.gold_answer,
            expect_abstention=question.expect_abstention,
        )
        if "citation_support" not in judged:
            judged["citation_support"] = DeterministicJudge().score(
                question=question.question,
                answer=result.answer,
                contexts=contexts,
                gold_answer=question.gold_answer,
                expect_abstention=question.expect_abstention,
            )["citation_support"]
        gold = set(question.gold_section_ids)
        row = {
            "id": question.id,
            "expect_abstention": question.expect_abstention,
            "abstained": is_abstention(result.answer),
            "recall_at_5": recall_at_k(gold, sections, 5) if gold else None,
            "recall_at_10": recall_at_k(gold, sections, 10) if gold else None,
            "mrr": mean_reciprocal_rank(gold, sections) if gold else None,
            "faithfulness": judged["faithfulness"],
            "answer_relevance": judged["answer_relevance"],
            "citation_support": judged["citation_support"],
            "latency_ms": result.latency_ms,
            "retrieved_section_ids": sections,
            "answer": result.answer,
        }
        rows.append(row)
    metrics = _aggregate(rows)
    return {
        "name": spec.name,
        "description": spec.description,
        "config": spec.model_dump(),
        "metrics": metrics.model_dump(),
        "questions": rows,
    }


def _aggregate(rows: list[dict]) -> Aggregate:
    answerable = [row for row in rows if row["recall_at_5"] is not None]
    abstain_rows = [row for row in rows if row["expect_abstention"]]

    def mean(values: list[float]) -> float:
        if not values:
            return 0.0
        return sum(values) / len(values)

    correct_abstentions = sum(1 for row in abstain_rows if row["abstained"])
    return Aggregate(
        n=len(rows),
        n_answerable=len(answerable),
        n_abstain=len(abstain_rows),
        recall_at_5=mean([row["recall_at_5"] for row in answerable]),
        recall_at_10=mean([row["recall_at_10"] for row in answerable]),
        mrr=mean([row["mrr"] for row in answerable]),
        faithfulness=mean([row["faithfulness"] for row in rows]),
        answer_relevance=mean([row["answer_relevance"] for row in rows]),
        citation_support=mean([row["citation_support"] for row in rows]),
        abstention_accuracy=(correct_abstentions / len(abstain_rows)) if abstain_rows else 0.0,
        mean_latency_ms=mean([row["latency_ms"] for row in rows]),
    )

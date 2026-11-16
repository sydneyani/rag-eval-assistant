"""End-to-end ingestion: load, chunk, embed, persist."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from citeground.config import Settings
from citeground.index.store import CorpusIndex, build_manifest
from citeground.ingest.chunking import Chunker, build_chunker
from citeground.ingest.loaders import load_corpus
from citeground.llm.base import Embedder
from citeground.llm.factory import build_embedder
from citeground.text import estimate_tokens


@dataclass(frozen=True)
class IngestResult:
    documents: int
    sections: int
    chunks: int
    index_dir: Path
    embedding_provider: str
    embedding_tokens: int


def ingest_corpus(
    settings: Settings,
    *,
    corpus_dir: Path | None = None,
    index_dir: Path | None = None,
    chunker: Chunker | None = None,
    embedder: Embedder | None = None,
) -> IngestResult:
    source = corpus_dir or settings.resolve(settings.corpus_dir)
    destination = index_dir or settings.resolve(settings.index_dir)
    reader = chunker or build_chunker(
        settings.chunker,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        sentence_window=settings.sentence_window,
        sentence_stride=settings.sentence_stride,
    )
    encoder = embedder or build_embedder(settings)
    documents = load_corpus(source)
    chunks = [chunk for document in documents for chunk in reader.chunk_document(document)]
    if not chunks:
        raise ValueError(f"Chunker produced no chunks from {source}")
    vectors = encoder.embed_documents([chunk.text for chunk in chunks])
    token_estimate = sum(estimate_tokens(chunk.text) for chunk in chunks)
    manifest = build_manifest(
        chunks=chunks,
        embedder_name=encoder.provider_name,
        embedder_model=encoder.model_name,
        dim=int(vectors.shape[1]),
        vector_store=settings.vector_store,
        chunker=reader.name,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        embedding_tokens=0 if encoder.provider_name == "hashing" else token_estimate,
    )
    CorpusIndex(chunks=chunks, vectors=vectors, manifest=manifest).save(destination)
    return IngestResult(
        documents=len(documents),
        sections=sum(len(document.sections) for document in documents),
        chunks=len(chunks),
        index_dir=destination,
        embedding_provider=encoder.provider_name,
        embedding_tokens=int(manifest["embedding_tokens_estimate"]),
    )

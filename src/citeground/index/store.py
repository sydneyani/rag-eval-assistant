"""Persist chunks and L2-normalized vectors.

The default dense backend is a NumPy matrix multiply. That is exact search,
which is the correct algorithm at the size of this corpus and it has no
native dependency. FAISS `IndexFlatIP` is available behind the same function
when the `vector` extra is installed: same scores, an index structure you
would actually keep once the corpus no longer fits in a single matrix
multiply you are willing to recompute. Approximate FAISS indexes are a later
swap once exact search shows up in the latency budget.

BM25 postings are rebuilt from chunk text on load. Rebuilding this corpus is
milliseconds. A persisted inverted index belongs in the design once ingestion
is incremental.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from citeground.errors import ConfigError, IndexNotFoundError
from citeground.schemas import Chunk

MANIFEST = "manifest.json"
CHUNKS = "chunks.jsonl"
VECTORS = "vectors.npy"


@dataclass
class CorpusIndex:
    chunks: list[Chunk]
    vectors: np.ndarray
    manifest: dict[str, object]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / CHUNKS).open("w", encoding="utf-8") as handle:
            for chunk in self.chunks:
                handle.write(chunk.model_dump_json() + "\n")
        np.save(directory / VECTORS, self.vectors.astype(np.float32))
        (directory / MANIFEST).write_text(json.dumps(self.manifest, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> CorpusIndex:
        manifest_path = directory / MANIFEST
        chunks_path = directory / CHUNKS
        vectors_path = directory / VECTORS
        if not (manifest_path.exists() and chunks_path.exists() and vectors_path.exists()):
            raise IndexNotFoundError(
                f"No index in {directory}. Build one with `python -m citeground.cli.ingest`."
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        chunks = [
            Chunk.model_validate_json(line)
            for line in chunks_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        vectors = np.load(vectors_path).astype(np.float32)
        if len(chunks) != vectors.shape[0]:
            raise IndexNotFoundError(
                f"Index in {directory} is corrupt: {len(chunks)} chunks, {vectors.shape[0]} vectors"
            )
        return cls(chunks=chunks, vectors=vectors, manifest=manifest)


def build_manifest(
    *,
    chunks: list[Chunk],
    embedder_name: str,
    embedder_model: str,
    dim: int,
    vector_store: str,
    chunker: str,
    chunk_size: int,
    chunk_overlap: int,
    embedding_tokens: int,
) -> dict[str, object]:
    doc_ids = sorted({chunk.doc_id for chunk in chunks})
    return {
        "version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "embedding_provider": embedder_name,
        "embedding_model": embedder_model,
        "embedding_dim": dim,
        "embedding_tokens_estimate": embedding_tokens,
        "vector_store": vector_store,
        "chunker": chunker,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "documents": len(doc_ids),
        "chunks": len(chunks),
        "doc_ids": doc_ids,
    }


def _normalize(query: np.ndarray) -> np.ndarray:
    vector = np.asarray(query, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if norm == 0:
        return vector
    return vector / np.float32(norm)


def dense_search_numpy(vectors: np.ndarray, query: np.ndarray, k: int) -> list[tuple[int, float]]:
    if k <= 0 or vectors.size == 0:
        return []
    query_vector = _normalize(query)
    if float(np.linalg.norm(query_vector)) == 0:
        return []
    scores = vectors @ query_vector
    count = min(k, int(scores.shape[0]))
    if count == scores.shape[0]:
        order = np.argsort(-scores)
    else:
        partition = np.argpartition(-scores, count - 1)[:count]
        order = partition[np.argsort(-scores[partition])]
    return [(int(index), float(scores[index])) for index in order[:count]]


def dense_search_faiss(vectors: np.ndarray, query: np.ndarray, k: int) -> list[tuple[int, float]]:
    if k <= 0 or vectors.size == 0:
        return []
    try:
        import faiss
    except ImportError as exc:
        raise ConfigError(
            "FAISS is not installed. Install the vector extra (`pip install -e '.[vector]'`) "
            "or set CITEGROUND_VECTOR_STORE=numpy."
        ) from exc
    matrix = np.ascontiguousarray(vectors.astype(np.float32))
    query_vector = _normalize(query).reshape(1, -1)
    if float(np.linalg.norm(query_vector)) == 0:
        return []
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    count = min(k, matrix.shape[0])
    scores, ids = index.search(query_vector, count)
    hits: list[tuple[int, float]] = []
    for score, row_id in zip(scores[0], ids[0], strict=True):
        if int(row_id) < 0:
            continue
        hits.append((int(row_id), float(score)))
    return hits


def dense_search(
    vectors: np.ndarray,
    query: np.ndarray,
    k: int,
    *,
    backend: str = "numpy",
) -> list[tuple[int, float]]:
    if backend == "numpy":
        return dense_search_numpy(vectors, query, k)
    if backend == "faiss":
        return dense_search_faiss(vectors, query, k)
    raise ValueError(f"Unknown vector store '{backend}'. Expected numpy or faiss.")

"""Build the local index from the markdown corpus."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from citeground.config import Settings
from citeground.ingest.pipeline import ingest_corpus


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="citeground-ingest",
        description="Chunk the corpus, embed it, and write a local index.",
    )
    parser.add_argument("--corpus", type=Path, default=None, help="Markdown corpus directory")
    parser.add_argument("--index", type=Path, default=None, help="Directory to write the index")
    parser.add_argument("--chunker", choices=["fixed", "recursive", "sentence"], default=None)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--overlap", type=int, default=None)
    parser.add_argument("--embedding", choices=["auto", "hashing", "openai"], default=None)
    parser.add_argument("--vector-store", choices=["numpy", "faiss"], default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    overrides: dict[str, object] = {}
    if args.chunker:
        overrides["chunker"] = args.chunker
    if args.chunk_size:
        overrides["chunk_size"] = args.chunk_size
    if args.overlap is not None:
        overrides["chunk_overlap"] = args.overlap
    if args.embedding:
        overrides["embedding_provider"] = args.embedding
    if args.vector_store:
        overrides["vector_store"] = args.vector_store
    settings = Settings(**overrides)
    result = ingest_corpus(
        settings,
        corpus_dir=args.corpus,
        index_dir=args.index,
    )
    print(
        f"Indexed {result.chunks} chunks from {result.documents} documents "
        f"({result.sections} sections) into {result.index_dir} "
        f"using the {result.embedding_provider} embedder."
    )


if __name__ == "__main__":
    main()

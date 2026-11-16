"""Chunking strategies.

Three strategies share one contract: they never cross a section boundary,
and every chunk carries the parent section id. That keeps retrieval
experiments comparable, because the labeled unit is the section.

* fixed — character windows with overlap. Predictable, and the baseline that
  shows what you give up by ignoring paragraph boundaries.
* recursive — pack paragraphs, falling back to sentences and then character
  windows when a unit is larger than the budget. This is the default.
* sentence — a sliding window of N sentences. Useful when answers are single
  statements and you want the window itself to be the citation.
"""

from __future__ import annotations

import re
from typing import Protocol

from citeground.schemas import Chunk, ParsedDocument, Section
from citeground.text import split_sentences, windows

_PARAGRAPHS = re.compile(r"\n\s*\n")


class Chunker(Protocol):
    name: str

    def chunk_document(self, document: ParsedDocument) -> list[Chunk]: ...


def _locate(section_text: str, snippet: str, cursor: int) -> tuple[int, int]:
    """Find a snippet in the section, tolerating whitespace differences."""

    needle = snippet.strip()
    if not needle:
        return cursor, cursor
    direct = section_text.find(needle, cursor)
    if direct >= 0:
        return direct, direct + len(needle)
    probe = needle[:80]
    found = section_text.find(probe, cursor)
    if found < 0:
        found = section_text.find(probe)
    if found < 0:
        return cursor, min(len(section_text), cursor + len(needle))
    return found, min(len(section_text), found + len(needle))


def _make_chunks(
    section: Section,
    *,
    doc_title: str,
    parts: list[str],
    metadata: dict[str, str],
    chunker_name: str,
    start_position: int,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    cursor = 0
    meta = {**metadata, "chunker": chunker_name}
    for part in parts:
        text = part.strip()
        if not text:
            continue
        start, end = _locate(section.text, text, cursor)
        cursor = start + 1
        position = start_position + len(chunks)
        chunks.append(
            Chunk(
                chunk_id=f"{section.section_id}#{position:03d}",
                doc_id=section.doc_id,
                section_id=section.section_id,
                doc_title=doc_title,
                section_title=section.title,
                text=text,
                position=position,
                start_char=start,
                end_char=end,
                metadata=meta,
            )
        )
    return chunks


class FixedChunker:
    name = "fixed"

    def __init__(self, size: int = 900, overlap: int = 150) -> None:
        if overlap < 0 or overlap >= size:
            raise ValueError("overlap must be >= 0 and < chunk size")
        self.size = size
        self.overlap = overlap

    def chunk_document(self, document: ParsedDocument) -> list[Chunk]:
        chunks: list[Chunk] = []
        for section in document.sections:
            spans = windows(section.text, self.size, self.overlap)
            parts = [section.text[start:end] for start, end in spans]
            chunks.extend(
                _make_chunks(
                    section,
                    doc_title=document.title,
                    parts=parts,
                    metadata=document.metadata,
                    chunker_name=self.name,
                    start_position=len(chunks),
                )
            )
        return chunks


def _atomic_pieces(text: str, size: int, overlap: int) -> list[str]:
    """Break text into pieces that each fit in `size` characters."""

    paragraphs = [part.strip() for part in _PARAGRAPHS.split(text) if part.strip()]
    if not paragraphs:
        paragraphs = [text.strip()] if text.strip() else []
    pieces: list[str] = []
    for paragraph in paragraphs:
        if len(paragraph) <= size:
            pieces.append(paragraph)
            continue
        sentences = split_sentences(paragraph)
        if len(sentences) <= 1:
            pieces.extend(paragraph[start:end] for start, end in windows(paragraph, size, overlap))
            continue
        buffer = ""
        for sentence in sentences:
            if len(sentence) > size:
                if buffer:
                    pieces.append(buffer)
                    buffer = ""
                spans = windows(sentence, size, overlap)
                pieces.extend(sentence[start:end] for start, end in spans)
                continue
            candidate = sentence if not buffer else f"{buffer} {sentence}"
            if len(candidate) <= size:
                buffer = candidate
            else:
                pieces.append(buffer)
                buffer = sentence
        if buffer:
            pieces.append(buffer)
    return pieces


def _joined_length(parts: list[str]) -> int:
    if not parts:
        return 0
    return sum(len(part) for part in parts) + 2 * (len(parts) - 1)


def _tail(parts: list[str], overlap: int) -> list[str]:
    """Keep a trailing run of pieces whose joined length is within `overlap`."""

    kept: list[str] = []
    for previous in reversed(parts):
        candidate = [previous, *kept]
        if _joined_length(candidate) > overlap:
            break
        kept.append(previous)
    return list(reversed(kept))


def pack_pieces(pieces: list[str], size: int, overlap: int) -> list[str]:
    """Greedy pack. The next chunk repeats a trailing suffix up to `overlap`.

    A single piece longer than `size` is emitted alone. Callers are expected
    to split those first; the guard is here so a bad piece cannot loop.
    """

    if not pieces:
        return []
    chunks: list[str] = []
    current: list[str] = []
    for piece in pieces:
        if current and _joined_length(current) + 2 + len(piece) > size:
            chunks.append("\n\n".join(current))
            current = _tail(current, overlap)
            while current and _joined_length(current) + 2 + len(piece) > size:
                current.pop(0)
        current.append(piece)
    if current:
        chunks.append("\n\n".join(current))
    return chunks


class RecursiveChunker:
    name = "recursive"

    def __init__(self, size: int = 900, overlap: int = 150) -> None:
        if overlap < 0 or overlap >= size:
            raise ValueError("overlap must be >= 0 and < chunk size")
        self.size = size
        self.overlap = overlap

    def chunk_document(self, document: ParsedDocument) -> list[Chunk]:
        chunks: list[Chunk] = []
        for section in document.sections:
            pieces = _atomic_pieces(section.text, self.size, self.overlap)
            parts = pack_pieces(pieces, self.size, self.overlap)
            chunks.extend(
                _make_chunks(
                    section,
                    doc_title=document.title,
                    parts=parts,
                    metadata=document.metadata,
                    chunker_name=self.name,
                    start_position=len(chunks),
                )
            )
        return chunks


class SentenceWindowChunker:
    name = "sentence"

    def __init__(self, window: int = 3, stride: int = 2) -> None:
        if window < 1:
            raise ValueError("sentence window must be >= 1")
        if stride < 1:
            raise ValueError("sentence stride must be >= 1")
        self.window = window
        self.stride = stride

    def chunk_document(self, document: ParsedDocument) -> list[Chunk]:
        chunks: list[Chunk] = []
        for section in document.sections:
            sentences = split_sentences(section.text)
            if not sentences:
                continue
            parts: list[str] = []
            start = 0
            while start < len(sentences):
                group = sentences[start : start + self.window]
                parts.append(" ".join(group))
                if start + self.window >= len(sentences):
                    break
                start += self.stride
            chunks.extend(
                _make_chunks(
                    section,
                    doc_title=document.title,
                    parts=parts,
                    metadata=document.metadata,
                    chunker_name=self.name,
                    start_position=len(chunks),
                )
            )
        return chunks


def build_chunker(
    name: str,
    *,
    chunk_size: int,
    chunk_overlap: int,
    sentence_window: int,
    sentence_stride: int,
) -> Chunker:
    if name == "fixed":
        return FixedChunker(chunk_size, chunk_overlap)
    if name == "recursive":
        return RecursiveChunker(chunk_size, chunk_overlap)
    if name == "sentence":
        return SentenceWindowChunker(sentence_window, sentence_stride)
    raise ValueError(f"Unknown chunker '{name}'. Expected fixed, recursive, or sentence.")

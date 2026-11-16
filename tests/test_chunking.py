from pathlib import Path

from citeground.ingest.chunking import FixedChunker, RecursiveChunker, SentenceWindowChunker
from citeground.ingest.loaders import parse_markdown


def _document(body: str, tmp_path: Path):
    path = tmp_path / "sample.md"
    raw = "---\ndoc_id: sample\ntitle: Sample Filing\nsource_type: synthetic-10k\n---\n" + body
    return parse_markdown(path, raw)


def test_loader_uses_explicit_section_ids_and_skips_empty_preamble(tmp_path: Path) -> None:
    document = _document(
        "\n# Item 1. Business\n\n## Revenue picture {results}\n\nFiscal 2024 revenue was $4.82 billion.\n",
        tmp_path,
    )
    assert document.doc_id == "sample"
    assert document.title == "Sample Filing"
    assert [section.section_id for section in document.sections] == ["sample/results"]
    assert document.metadata["source_type"] == "synthetic-10k"


def test_duplicate_section_slugs_are_disambiguated(tmp_path: Path) -> None:
    document = _document(
        "\n## Notes {notes}\n\nFirst.\n\n## Notes {notes}\n\nSecond.\n",
        tmp_path,
    )
    assert [section.section_id for section in document.sections] == [
        "sample/notes",
        "sample/notes-2",
    ]


def test_fixed_chunker_does_not_cross_sections(tmp_path: Path) -> None:
    body = (
        "\n## Alpha {alpha}\n\n"
        + ("Alpha fact. " * 200)
        + "\n\n## Beta {beta}\n\n"
        + ("Beta fact. " * 40)
    )
    document = _document(body, tmp_path)
    chunks = FixedChunker(size=500, overlap=50).chunk_document(document)
    alpha = [chunk for chunk in chunks if chunk.section_id == "sample/alpha"]
    beta = [chunk for chunk in chunks if chunk.section_id == "sample/beta"]
    assert len(alpha) > 1
    assert beta
    assert all("Beta" not in chunk.text for chunk in alpha)
    assert all("Alpha" not in chunk.text for chunk in beta)
    assert all(chunk.metadata["chunker"] == "fixed" for chunk in chunks)


def test_recursive_chunker_packs_short_paragraphs(tmp_path: Path) -> None:
    body = "\n## Facts {facts}\n\n" + "\n\n".join(
        f"Paragraph {index} is short." for index in range(6)
    )
    document = _document(body, tmp_path)
    chunks = RecursiveChunker(size=200, overlap=0).chunk_document(document)
    assert chunks
    assert all(len(chunk.text) <= 200 for chunk in chunks)
    joined = "\n".join(chunk.text for chunk in chunks)
    for index in range(6):
        assert f"Paragraph {index}" in joined


def test_sentence_windows_overlap_by_stride(tmp_path: Path) -> None:
    sentences = " ".join(f"Sentence {index} is here." for index in range(5))
    document = _document(f"\n## Story {sentences}\n\n{sentences}\n", tmp_path)
    # The heading has no explicit id; the body is the sentences.
    chunks = SentenceWindowChunker(window=3, stride=2).chunk_document(document)
    assert len(chunks) >= 2
    assert "Sentence 0" in chunks[0].text and "Sentence 2" in chunks[0].text
    assert "Sentence 2" in chunks[1].text


def test_chunk_ids_are_stable(tmp_path: Path) -> None:
    document = _document("\n## One {one}\n\nHello filing world.\n", tmp_path)
    first = RecursiveChunker().chunk_document(document)
    second = RecursiveChunker().chunk_document(document)
    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]

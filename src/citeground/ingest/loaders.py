"""Markdown corpus loader.

Documents are Markdown with a tiny `key: value` front matter block. Sections
are `##` headings. A heading may set a stable id with a trailing `{slug}` so
evaluation labels do not depend on heading wording.

The loader intentionally does not depend on a YAML parser. Front matter for
this corpus is a flat string map, and a one-purpose parser is easier to
audit than a transitive YAML dependency.
"""

from __future__ import annotations

import re
from pathlib import Path

from citeground.schemas import ParsedDocument, Section

_FRONT_MATTER = re.compile(r"\A---\r?\n(.*?)\r?\n---\r?\n", re.DOTALL)
_EXPLICIT_ID = re.compile(r"^(.*?)\s*\{([a-z0-9_-]+)\}\s*$")
_SKIP_NAMES = {"readme.md", "corpus.md"}


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:80] or "section"


def _parse_front_matter(raw: str) -> tuple[dict[str, str], str]:
    match = _FRONT_MATTER.match(raw)
    if not match:
        return {}, raw
    meta: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if not line.strip() or ":" not in line:
            continue
        key, value = line.split(":", 1)
        meta[key.strip()] = value.strip().strip('"')
    return meta, raw[match.end() :]


def parse_markdown(path: Path, raw: str | None = None) -> ParsedDocument:
    text = raw if raw is not None else path.read_text(encoding="utf-8")
    meta, body = _parse_front_matter(text)
    doc_id = meta.get("doc_id") or slugify(path.stem)
    title = meta.get("title") or path.stem
    sections = _split_sections(doc_id, body)
    return ParsedDocument(
        doc_id=doc_id,
        title=title,
        source_path=str(path),
        metadata={key: value for key, value in meta.items() if key not in {"doc_id", "title"}},
        sections=sections,
    )


def _split_sections(doc_id: str, body: str) -> list[Section]:
    sections: list[Section] = []
    item_heading: str | None = None
    current_title = "Preamble"
    current_slug = "preamble"
    buffer: list[str] = []
    seen: dict[str, int] = {}

    def flush() -> None:
        text = "\n".join(buffer).strip()
        if not text:
            return
        slug = current_slug
        seen[slug] = seen.get(slug, 0) + 1
        if seen[slug] > 1:
            slug = f"{slug}-{seen[slug]}"
        heading_path = [part for part in [item_heading, current_title] if part]
        sections.append(
            Section(
                doc_id=doc_id,
                section_id=f"{doc_id}/{slug}",
                title=current_title,
                text=text,
                heading_path=heading_path,
            )
        )

    for line in body.splitlines():
        if line.startswith("# ") and not line.startswith("## "):
            flush()
            buffer = []
            item_heading = line[2:].strip()
            current_title = "Preamble"
            current_slug = "preamble"
            continue
        if line.startswith("## "):
            flush()
            buffer = []
            raw_heading = line[3:].strip()
            explicit = _EXPLICIT_ID.match(raw_heading)
            if explicit:
                current_title = explicit.group(1).strip()
                current_slug = explicit.group(2)
            else:
                current_title = raw_heading
                current_slug = slugify(raw_heading)
            continue
        buffer.append(line)
    flush()
    return sections


def load_corpus(corpus_dir: Path) -> list[ParsedDocument]:
    if not corpus_dir.is_dir():
        raise FileNotFoundError(f"Corpus directory does not exist: {corpus_dir}")
    documents: list[ParsedDocument] = []
    for path in sorted(corpus_dir.glob("*.md")):
        if path.name.lower() in _SKIP_NAMES:
            continue
        documents.append(parse_markdown(path))
    if not documents:
        raise FileNotFoundError(f"No markdown documents found in {corpus_dir}")
    return documents

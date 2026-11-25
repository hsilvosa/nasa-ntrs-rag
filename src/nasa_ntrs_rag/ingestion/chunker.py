"""Chunking strategies: Baseline Fixed-Size (v0) and Hierarchical Parent-Child (v1+)."""

from __future__ import annotations

import re
from dataclasses import dataclass

from nasa_ntrs_rag.schemas import ChildChunk, NTRSDocumentMetadata, ParentChunk

_PAGE_MARKER_RE = re.compile(r"<!--\s*page:\s*(\d+)\s*-->")
_HEADING_LINE_RE = re.compile(r"^(#{1,4})\s+(.+?)\s*$")


@dataclass
class _SectionBlock:
    section_title: str
    text: str
    page_start: int | None
    page_end: int | None


def _strip_page_markers(text: str) -> str:
    return _PAGE_MARKER_RE.sub("", text).strip()


def _extract_sections(markdown_text: str) -> list[_SectionBlock]:
    """Split Markdown by headings while tracking page ranges via <!-- page: N --> markers."""
    sections: list[_SectionBlock] = []
    current_title = "Document Overview"
    current_lines: list[str] = []
    current_page: int | None = None
    sec_page_start: int | None = None
    sec_page_end: int | None = None

    def flush() -> None:
        nonlocal current_lines, sec_page_start, sec_page_end
        body = "\n".join(current_lines).strip()
        if body:
            sections.append(
                _SectionBlock(
                    section_title=current_title,
                    text=body,
                    page_start=sec_page_start,
                    page_end=sec_page_end or sec_page_start,
                )
            )
        current_lines = []
        sec_page_start = current_page
        sec_page_end = current_page

    for line in markdown_text.splitlines():
        page_match = _PAGE_MARKER_RE.search(line)
        if page_match:
            current_page = int(page_match.group(1))
            if sec_page_start is None:
                sec_page_start = current_page
            sec_page_end = current_page
            cleaned_line = _PAGE_MARKER_RE.sub("", line).strip()
            if not cleaned_line:
                continue
            line = cleaned_line

        heading_match = _HEADING_LINE_RE.match(line.strip())
        if heading_match:
            flush()
            current_title = heading_match.group(2).strip()
            sec_page_start = current_page
            sec_page_end = current_page
        else:
            if sec_page_start is None and current_page is not None:
                sec_page_start = current_page
            if current_page is not None:
                sec_page_end = current_page
            current_lines.append(line)

    flush()
    return sections


def _split_preserving_tables(text: str, max_chars: int, overlap: int) -> list[str]:
    """Split text into chunks without breaking Markdown tables mid-row."""
    paragraphs = re.split(r"\n{2,}", text.strip())
    chunks: list[str] = []
    buffer: list[str] = []
    buffer_len = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        is_table = para.startswith("|") and "\n|" in para
        # If a single non-table paragraph exceeds max_chars, hard-slice it with overlap
        if len(para) > max_chars and not is_table:
            if buffer:
                chunks.append("\n\n".join(buffer))
                buffer = []
                buffer_len = 0
            step = max(1, max_chars - overlap)
            for start in range(0, len(para), step):
                piece = para[start : start + max_chars].strip()
                if piece:
                    chunks.append(piece)
            continue

        projected = buffer_len + len(para) + (2 if buffer else 0)
        if buffer and projected > max_chars:
            chunks.append("\n\n".join(buffer))
            # Keep trailing overlap context from previous buffer when feasible
            tail = buffer[-1] if overlap > 0 and len(buffer[-1]) <= overlap * 2 else ""
            buffer = [tail, para] if tail else [para]
            buffer_len = sum(len(x) for x in buffer) + (2 if len(buffer) > 1 else 0)
        else:
            buffer.append(para)
            buffer_len = projected

    if buffer:
        chunks.append("\n\n".join(buffer))
    return chunks


def _build_contextual_header(doc: NTRSDocumentMetadata, section_title: str) -> str:
    """Create a concise contextual prefix injected before child text for embedding."""
    year = doc.publication_year or "Unknown"
    center = doc.center or "NASA"
    return (
        f"[NASA NTRS {doc.doc_id} | {doc.title} | Year: {year} | "
        f"Center: {center} | Section: {section_title}]"
    )


def hierarchical_chunk_document(
    doc: NTRSDocumentMetadata,
    markdown_text: str,
    parent_chars: int = 4800,
    child_chars: int = 1200,
    child_overlap: int = 200,
) -> tuple[list[ParentChunk], list[ChildChunk]]:
    """Build Parent-Child chunks with section awareness and contextual headers (v1+)."""
    sections = _extract_sections(markdown_text)
    parents: list[ParentChunk] = []
    children: list[ChildChunk] = []

    base_meta = {
        "publication_year": doc.publication_year,
        "center": doc.center,
        "sti_type": doc.sti_type,
        "subject_categories": doc.subject_categories,
        "topic_query": doc.topic_query,
        "ntrs_url": doc.ntrs_url,
    }

    child_counter = 0
    parent_counter = 0

    for sec in sections:
        parent_texts = _split_preserving_tables(sec.text, max_chars=parent_chars, overlap=0)
        for p_text in parent_texts:
            parent_id = f"{doc.doc_id}::p{parent_counter:04d}"
            parent_counter += 1
            parents.append(
                ParentChunk(
                    parent_id=parent_id,
                    doc_id=doc.doc_id,
                    title=doc.title,
                    section_title=sec.section_title,
                    text=p_text,
                    page_start=sec.page_start,
                    page_end=sec.page_end,
                    metadata=base_meta,
                )
            )

            header = _build_contextual_header(doc, sec.section_title)
            child_texts = _split_preserving_tables(
                p_text, max_chars=child_chars, overlap=child_overlap
            )
            for c_text in child_texts:
                chunk_id = f"{doc.doc_id}::c{child_counter:05d}"
                children.append(
                    ChildChunk(
                        chunk_id=chunk_id,
                        parent_id=parent_id,
                        doc_id=doc.doc_id,
                        title=doc.title,
                        section_title=sec.section_title,
                        text=c_text,
                        contextual_text=f"{header}\n{c_text}",
                        chunk_index=child_counter,
                        page_start=sec.page_start,
                        page_end=sec.page_end,
                        metadata=base_meta,
                    )
                )
                child_counter += 1

    return parents, children


def fixed_chunk_document(
    doc: NTRSDocumentMetadata,
    raw_text: str,
    chunk_chars: int = 1000,
    overlap: int = 150,
) -> tuple[list[ParentChunk], list[ChildChunk]]:
    """Naive fixed-window chunker used strictly for the v0 baseline ablation."""
    clean = _strip_page_markers(raw_text)
    step = max(1, chunk_chars - overlap)
    parents: list[ParentChunk] = []
    children: list[ChildChunk] = []

    base_meta = {
        "publication_year": doc.publication_year,
        "center": doc.center,
        "subject_categories": doc.subject_categories,
        "ntrs_url": doc.ntrs_url,
    }

    for idx, start in enumerate(range(0, len(clean), step)):
        piece = clean[start : start + chunk_chars].strip()
        if not piece:
            continue
        parent_id = f"{doc.doc_id}::v0p{idx:05d}"
        chunk_id = f"{doc.doc_id}::v0c{idx:05d}"
        parents.append(
            ParentChunk(
                parent_id=parent_id,
                doc_id=doc.doc_id,
                title=doc.title,
                section_title="Unstructured",
                text=piece,
                metadata=base_meta,
            )
        )
        children.append(
            ChildChunk(
                chunk_id=chunk_id,
                parent_id=parent_id,
                doc_id=doc.doc_id,
                title=doc.title,
                section_title="Unstructured",
                text=piece,
                contextual_text=piece,
                chunk_index=idx,
                metadata=base_meta,
            )
        )

    return parents, children

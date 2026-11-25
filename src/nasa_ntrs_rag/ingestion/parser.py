"""Layout-aware PDF to Markdown parser supporting Docling and PyMuPDF backends."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

ParserBackend = Literal["auto", "docling", "pymupdf", "plain"]

_HEADING_RE = re.compile(
    r"^(?:\d+(?:\.\d+)*\.?\s+[A-Z][A-Za-z0-9 ,:/\-()]{2,80}|"
    r"(?:ABSTRACT|INTRODUCTION|BACKGROUND|METHODOLOGY|RESULTS|DISCUSSION|"
    r"CONCLUSIONS?|RECOMMENDATIONS?|REFERENCES|APPENDIX(?:\s+[A-Z0-9]+)?))$",
    re.MULTILINE,
)


def parse_pdf_to_markdown(
    pdf_path: Path,
    backend: ParserBackend = "auto",
) -> str:
    """Convert a technical PDF into structured Markdown with <!-- page: N --> markers.

    Backends:
    - ``docling``: High-fidelity layout/table extraction (requires optional `docling` package).
    - ``pymupdf``: Fast PyMuPDF block extraction with table detection and heading heuristics.
    - ``plain``: Raw text dump for v0 baseline ablation experiments.
    - ``auto``: Uses Docling when installed, otherwise falls back to PyMuPDF.
    """
    if backend == "auto":
        try:
            return _parse_with_docling(pdf_path)
        except ImportError:
            logger.info("Docling not installed; falling back to PyMuPDF parser.")
            return _parse_with_pymupdf(pdf_path, detect_tables=True)
    if backend == "docling":
        return _parse_with_docling(pdf_path)
    if backend == "pymupdf":
        return _parse_with_pymupdf(pdf_path, detect_tables=True)
    if backend == "plain":
        return _parse_with_pymupdf(pdf_path, detect_tables=False)
    raise ValueError(f"Unsupported parser backend: {backend}")


def _parse_with_docling(pdf_path: Path) -> str:
    """Parse PDF with Docling preserving layout, section hierarchy, and Markdown tables."""
    from docling.document_converter import DocumentConverter  # type: ignore[import-not-found]

    converter = DocumentConverter()
    result = converter.convert(str(pdf_path))
    doc = result.document
    # Export page-by-page when available so exact page numbers survive into chunk metadata
    num_pages = len(getattr(doc, "pages", {}) or {})
    if num_pages > 0:
        parts: list[str] = []
        for page_no in range(1, num_pages + 1):
            page_md = doc.export_to_markdown(page_no=page_no).strip()
            if page_md:
                parts.append(f"<!-- page: {page_no} -->\n{page_md}")
        if parts:
            return "\n\n".join(parts)
    return f"<!-- page: 1 -->\n{doc.export_to_markdown().strip()}"


def _format_table_as_markdown(rows: list[list[str | None]]) -> str:
    """Format extracted table cells into a valid GitHub-Flavored Markdown table."""
    cleaned = [
        [re.sub(r"\s+", " ", (cell or "").strip()) for cell in row]
        for row in rows
        if any(cell and cell.strip() for cell in row)
    ]
    if not cleaned:
        return ""
    width = max(len(r) for r in cleaned)
    padded = [r + [""] * (width - len(r)) for r in cleaned]
    header = "| " + " | ".join(padded[0]) + " |"
    sep = "| " + " | ".join(["---"] * width) + " |"
    body = ["| " + " | ".join(row) + " |" for row in padded[1:]]
    return "\n".join([header, sep, *body])


def _promote_headings(text: str) -> str:
    """Promote NASA report numbered/uppercase section titles into Markdown headings."""
    lines = text.splitlines()
    out: list[str] = []
    for line in lines:
        stripped = line.strip()
        if (
            stripped
            and not stripped.startswith("#")
            and not stripped.startswith("|")
            and _HEADING_RE.match(stripped)
        ):
            out.append(f"## {stripped}")
        else:
            out.append(line)
    return "\n".join(out)


def _parse_with_pymupdf(pdf_path: Path, detect_tables: bool = True) -> str:
    """Parse PDF with PyMuPDF, optionally extracting tables to Markdown."""
    import pymupdf

    parts: list[str] = []
    with pymupdf.open(pdf_path) as doc:
        for page_idx, page in enumerate(doc, start=1):
            raw_text = page.get_text("text") or ""
            cleaned = re.sub(r"\n{3,}", "\n\n", raw_text).strip()
            if detect_tables:
                cleaned = _promote_headings(cleaned)
                try:
                    tabs = page.find_tables()
                    md_tables: list[str] = []
                    for tab in tabs.tables:
                        md_tbl = _format_table_as_markdown(tab.extract())
                        if md_tbl:
                            md_tables.append(md_tbl)
                    if md_tables:
                        cleaned = cleaned + "\n\n" + "\n\n".join(md_tables)
                except Exception:
                    # Table detection is best-effort on older/scanned PDFs
                    pass
            if cleaned:
                parts.append(f"<!-- page: {page_idx} -->\n{cleaned}")
    return "\n\n".join(parts)

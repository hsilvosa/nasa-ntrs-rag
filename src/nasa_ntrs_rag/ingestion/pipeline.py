"""Offline ingestion pipeline: parse NASA NTRS PDFs to Markdown and build parent-child chunks."""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Literal

from nasa_ntrs_rag.config import Settings, get_settings
from nasa_ntrs_rag.ingestion.chunker import fixed_chunk_document, hierarchical_chunk_document
from nasa_ntrs_rag.ingestion.ntrs_client import load_manifest
from nasa_ntrs_rag.ingestion.parser import ParserBackend, parse_pdf_to_markdown
from nasa_ntrs_rag.schemas import ChildChunk, ParentChunk

logger = logging.getLogger(__name__)

ChunkingStrategy = Literal["hierarchical", "fixed"]


def save_chunks(
    parents: list[ParentChunk],
    children: list[ChildChunk],
    out_dir: Path,
    prefix: str = "",
) -> tuple[Path, Path]:
    """Serialize parent and child chunks to JSONL files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    p_name = f"{prefix}parents.jsonl" if prefix else "parents.jsonl"
    c_name = f"{prefix}children.jsonl" if prefix else "children.jsonl"
    p_path = out_dir / p_name
    c_path = out_dir / c_name

    with p_path.open("w", encoding="utf-8") as fh:
        for p in parents:
            fh.write(p.model_dump_json() + "\n")
    with c_path.open("w", encoding="utf-8") as fh:
        for c in children:
            fh.write(c.model_dump_json() + "\n")
    return p_path, c_path


def load_chunks(
    chunks_dir: Path, prefix: str = ""
) -> tuple[dict[str, ParentChunk], list[ChildChunk]]:
    """Load parent lookup map and child chunk list from JSONL."""
    p_path = chunks_dir / (f"{prefix}parents.jsonl" if prefix else "parents.jsonl")
    c_path = chunks_dir / (f"{prefix}children.jsonl" if prefix else "children.jsonl")

    parents: dict[str, ParentChunk] = {}
    if p_path.exists():
        with p_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    obj = ParentChunk.model_validate_json(line)
                    parents[obj.parent_id] = obj

    children: list[ChildChunk] = []
    if c_path.exists():
        with c_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    children.append(ChildChunk.model_validate_json(line))
    return parents, children


def _parse_single_pdf_worker(
    doc_id: str,
    title: str,
    pdf_path_str: str,
    md_path_str: str,
    parser_backend: ParserBackend,
) -> tuple[str, bool, str | None]:
    """Top-level worker function for parallel PDF -> Markdown extraction."""
    try:
        markdown_text = parse_pdf_to_markdown(Path(pdf_path_str), backend=parser_backend)
        Path(md_path_str).write_text(markdown_text, encoding="utf-8")
        return doc_id, True, None
    except Exception as exc:
        return doc_id, False, f"{title[:40]}: {exc}"


def run_ingestion_pipeline(
    settings: Settings | None = None,
    parser_backend: ParserBackend = "auto",
    strategy: ChunkingStrategy = "hierarchical",
    force_reparse: bool = False,
    workers: int = 1,
) -> tuple[list[ParentChunk], list[ChildChunk]]:
    """Parse all PDFs listed in manifest.jsonl and produce parent/child chunk artifacts."""
    settings = settings or get_settings()
    settings.ensure_directories()
    docs = load_manifest(settings.manifest_path)
    if not docs:
        raise FileNotFoundError(
            f"No documents found in {settings.manifest_path}. Run download_corpus first."
        )

    md_suffix = "_plain.md" if parser_backend == "plain" else ".md"
    to_parse: list[tuple[str, str, Path, Path]] = []
    for doc in docs:
        pdf_path = Path(doc.pdf_path) if doc.pdf_path else settings.raw_pdf_dir / f"{doc.doc_id}.pdf"
        if not pdf_path.exists():
            logger.warning("PDF missing for doc_id=%s at %s", doc.doc_id, pdf_path)
            continue
        md_path = settings.processed_md_dir / f"{doc.doc_id}{md_suffix}"
        if force_reparse or not md_path.exists():
            to_parse.append((doc.doc_id, doc.title, pdf_path, md_path))

    if to_parse:
        if workers > 1:
            logger.info(
                "Parsing %d PDFs with %d worker processes (backend=%s)",
                len(to_parse),
                workers,
                parser_backend,
            )
            done = 0
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = [
                    pool.submit(
                        _parse_single_pdf_worker,
                        doc_id,
                        title,
                        str(pdf_path),
                        str(md_path),
                        parser_backend,
                    )
                    for doc_id, title, pdf_path, md_path in to_parse
                ]
                for fut in as_completed(futures):
                    doc_id, ok, err = fut.result()
                    done += 1
                    if not ok:
                        logger.warning("Failed to parse doc_id=%s (%s)", doc_id, err)
                    if done % 25 == 0 or done == len(to_parse):
                        logger.info("Parsed %d / %d PDFs", done, len(to_parse))
        else:
            for doc_id, title, pdf_path, md_path in to_parse:
                logger.info("Parsing %s (%s) with backend=%s", doc_id, title[:50], parser_backend)
                try:
                    markdown_text = parse_pdf_to_markdown(pdf_path, backend=parser_backend)
                    md_path.write_text(markdown_text, encoding="utf-8")
                except Exception as exc:
                    logger.warning("Failed to parse doc_id=%s: %s", doc_id, exc)

    all_parents: list[ParentChunk] = []
    all_children: list[ChildChunk] = []

    for doc in docs:
        md_path = settings.processed_md_dir / f"{doc.doc_id}{md_suffix}"
        if not md_path.exists():
            continue
        markdown_text = md_path.read_text(encoding="utf-8")
        if not markdown_text.strip():
            logger.warning("Empty text extracted for doc_id=%s", doc.doc_id)
            continue

        if strategy == "hierarchical":
            parents, children = hierarchical_chunk_document(
                doc=doc,
                markdown_text=markdown_text,
                parent_chars=settings.parent_chunk_chars,
                child_chars=settings.child_chunk_chars,
                child_overlap=settings.child_chunk_overlap,
            )
        else:
            parents, children = fixed_chunk_document(doc=doc, raw_text=markdown_text)

        all_parents.extend(parents)
        all_children.extend(children)

    prefix = "v0_" if strategy == "fixed" else ""
    save_chunks(all_parents, all_children, settings.processed_chunks_dir, prefix=prefix)
    logger.info(
        "Ingestion complete (%s): %d docs -> %d parents, %d children",
        strategy,
        len(docs),
        len(all_parents),
        len(all_children),
    )
    return all_parents, all_children

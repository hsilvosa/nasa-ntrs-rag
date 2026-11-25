"""Command-line interface for downloading, ingesting, indexing, querying, and benchmarking."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated, Literal

import typer
from rich.console import Console
from rich.panel import Panel

from nasa_ntrs_rag.config import get_settings
from nasa_ntrs_rag.evaluation.metrics import (
    build_starter_eval_from_manifest,
    format_ablation_markdown,
    load_eval_dataset,
    run_retrieval_ablation,
)
from nasa_ntrs_rag.generation.rag_pipeline import NASANTRSRAGPipeline
from nasa_ntrs_rag.ingestion.ntrs_client import DEFAULT_SAFETY_TOPICS, NTRSClient
from nasa_ntrs_rag.ingestion.parser import ParserBackend
from nasa_ntrs_rag.ingestion.pipeline import ChunkingStrategy, run_ingestion_pipeline
from nasa_ntrs_rag.retrieval.hybrid_store import HybridRetriever

app = typer.Typer(help="NASA NTRS Aerospace Safety RAG CLI")
console = Console()


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


@app.command()
def download(
    per_topic: Annotated[int, typer.Option(help="Max PDFs per aerospace safety topic.")] = 5,
    max_docs: Annotated[int, typer.Option(help="Total cap on downloaded NASA NTRS PDFs.")] = 30,
    topic: Annotated[list[str] | None, typer.Option(help="Custom search topics.")] = None,
) -> None:
    """Download NASA NTRS technical report PDFs and metadata into DATA_DIR."""
    _configure_logging()
    settings = get_settings()
    topics = topic or DEFAULT_SAFETY_TOPICS
    console.print(f"[cyan]Downloading NASA NTRS corpus into:[/cyan] {settings.data_dir}")
    with NTRSClient(settings) as client:
        docs = client.build_corpus(topics=topics, per_topic=per_topic, max_total_docs=max_docs)
    console.print(f"[green]Done.[/green] Total documents in manifest: {len(docs)}")


@app.command()
def ingest(
    backend: Annotated[
        ParserBackend, typer.Option(help="Parser backend: auto | docling | pymupdf | plain")
    ] = "auto",
    strategy: Annotated[
        ChunkingStrategy, typer.Option(help="Chunking strategy: hierarchical | fixed")
    ] = "hierarchical",
    force: Annotated[
        bool, typer.Option(help="Re-parse PDFs even if cached Markdown exists.")
    ] = False,
) -> None:
    """Parse downloaded PDFs to Markdown and generate Parent-Child (or v0 fixed) chunks."""
    _configure_logging()
    settings = get_settings()
    parents, children = run_ingestion_pipeline(
        settings=settings,
        parser_backend=backend,
        strategy=strategy,
        force_reparse=force,
    )
    console.print(
        f"[green]Ingestion complete[/green] ({strategy}): "
        f"{len(parents)} parent chunks, {len(children)} child chunks."
    )


@app.command()
def index(
    strategy: Annotated[
        ChunkingStrategy, typer.Option(help="Which chunk set to index.")
    ] = "hierarchical",
    collection: Annotated[
        str | None, typer.Option(help="Override Qdrant collection name.")
    ] = None,
) -> None:
    """Embed child chunks on GPU/CPU and build the Qdrant + BM25 hybrid index."""
    _configure_logging()
    settings = get_settings()
    prefix = "v0_" if strategy == "fixed" else ""
    default_col = (
        f"{settings.qdrant_collection}_v0" if strategy == "fixed" else settings.qdrant_collection
    )
    col_name = collection or default_col
    retriever = HybridRetriever(settings=settings, collection_name=col_name, chunks_prefix=prefix)
    count = retriever.index_chunks(recreate=True)
    console.print(f"[green]Indexed {count} chunks into collection '{col_name}'.[/green]")


@app.command()
def ask(
    question: Annotated[str, typer.Argument(help="Question to ask the NASA NTRS corpus.")],
    mode: Annotated[
        Literal["dense", "hybrid", "hybrid_rerank"], typer.Option(help="Retrieval mode.")
    ] = "hybrid_rerank",
    top_k: Annotated[int, typer.Option(help="Number of final sources to pass to LLM.")] = 5,
) -> None:
    """Run an end-to-end grounded RAG query from the terminal."""
    _configure_logging()
    pipeline = NASANTRSRAGPipeline(settings=get_settings())
    resp = pipeline.ask(query=question, mode=mode, top_k=top_k)
    console.print(Panel(resp.answer, title=f"Answer ({resp.mode}) | Abstained={resp.abstained}"))
    for cit in resp.citations:
        console.print(
            f"  [{cit.citation_index}] NTRS {cit.doc_id} | {cit.title} | "
            f"{cit.section_title} (p.{cit.page_start}) | score={cit.score}"
        )
    console.print(f"[dim]Timings: {resp.timings_ms.model_dump()}[/dim]")


@app.command("eval-ablation")
def eval_ablation(
    eval_file: Annotated[
        Path | None, typer.Option(help="Path to eval JSONL. Auto-generated if missing.")
    ] = None,
    max_factual: Annotated[
        int, typer.Option(help="Number of factual queries to auto-seed if eval file is missing.")
    ] = 30,
) -> None:
    """Run reproducible Dense vs Hybrid vs Hybrid+Rerank retrieval ablation benchmark."""
    _configure_logging()
    settings = get_settings()
    eval_path = eval_file or (settings.eval_dir / "starter_eval.jsonl")
    if not eval_path.exists():
        console.print(f"[yellow]Generating starter eval dataset at {eval_path}...[/yellow]")
        samples = build_starter_eval_from_manifest(
            manifest_path=settings.manifest_path,
            output_path=eval_path,
            max_factual=max_factual,
        )
    else:
        samples = load_eval_dataset(eval_path)

    retriever = HybridRetriever(settings=settings)
    out_json = settings.eval_dir / "ablation_results.json"
    report = run_retrieval_ablation(
        retriever=retriever,
        samples=samples,
        abstention_threshold=settings.abstention_score_threshold,
        output_json_path=out_json,
    )
    console.print("\n" + format_ablation_markdown(report))
    console.print(f"\n[green]Saved full ablation report to {out_json}[/green]")


@app.command()
def serve(
    host: Annotated[str, typer.Option()] = "127.0.0.1",
    port: Annotated[int, typer.Option()] = 8001,
) -> None:
    """Launch the FastAPI server and built-in RAG Inspector UI."""
    import uvicorn

    uvicorn.run("nasa_ntrs_rag.api.main:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    app()

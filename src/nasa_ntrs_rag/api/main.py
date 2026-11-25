"""FastAPI backend for NASA NTRS RAG with live Source & Score Inspector UI."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from nasa_ntrs_rag.config import get_settings
from nasa_ntrs_rag.generation.rag_pipeline import NASANTRSRAGPipeline
from nasa_ntrs_rag.schemas import RAGResponse

_INSPECTOR_TEMPLATE = Path(__file__).with_name("inspector.html")

app = FastAPI(
    title="NASA NTRS Aerospace Safety RAG",
    version="0.1.0",
    description=(
        "Hybrid Dense + BM25 + Cross-Encoder Reranking RAG over NASA NTRS Technical Reports"
    ),
)


@lru_cache(maxsize=1)
def get_pipeline() -> NASANTRSRAGPipeline:
    return NASANTRSRAGPipeline(settings=get_settings())


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3)
    mode: Literal["dense", "hybrid", "hybrid_rerank"] = "hybrid_rerank"
    top_k: int = Field(default=5, ge=1, le=20)
    use_metadata_filters: bool = True


@app.get("/health")
def health() -> dict[str, str]:
    settings = get_settings()
    return {
        "status": "ok",
        "data_dir": str(settings.data_dir),
        "device": settings.resolve_device(),
        "llm_provider": settings.llm_provider,
        "llm_model": settings.llm_model,
    }


@app.get("/stats")
def stats() -> dict[str, object]:
    pipeline = get_pipeline()
    return {
        "collection": pipeline.retriever.collection_name,
        "parent_chunks": len(pipeline.retriever.parents),
        "child_chunks": len(pipeline.retriever.children),
        "embedding_model": pipeline.settings.embedding_model,
        "reranker_model": pipeline.settings.reranker_model,
    }


@app.post("/query", response_model=RAGResponse)
def query_rag(req: QueryRequest) -> RAGResponse:
    try:
        return get_pipeline().ask(
            query=req.query,
            mode=req.mode,
            top_k=req.top_k,
            use_metadata_filters=req.use_metadata_filters,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/", response_class=HTMLResponse)
def inspector_ui() -> HTMLResponse:
    return HTMLResponse(content=_INSPECTOR_TEMPLATE.read_text(encoding="utf-8"))

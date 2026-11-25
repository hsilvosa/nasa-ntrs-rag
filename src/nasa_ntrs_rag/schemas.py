"""Shared Pydantic schemas for ingestion, retrieval, generation, and evaluation."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class NTRSDocumentMetadata(BaseModel):
    """Normalized metadata for a NASA NTRS technical document."""

    doc_id: str
    title: str
    abstract: str = ""
    publication_year: int | None = None
    center: str | None = None
    sti_type: str | None = None
    subject_categories: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    authors: list[str] = Field(default_factory=list)
    topic_query: str = ""
    pdf_url: str | None = None
    pdf_path: str | None = None
    ntrs_url: str = ""


class ParentChunk(BaseModel):
    """Broad section-level chunk sent to the LLM for rich causal context."""

    parent_id: str
    doc_id: str
    title: str
    section_title: str
    text: str
    page_start: int | None = None
    page_end: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChildChunk(BaseModel):
    """Fine-grained chunk indexed for high-precision dense + sparse retrieval."""

    chunk_id: str
    parent_id: str
    doc_id: str
    title: str
    section_title: str
    text: str
    contextual_text: str
    chunk_index: int
    page_start: int | None = None
    page_end: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryFilters(BaseModel):
    """Structured metadata filters extracted automatically from a user query."""

    year_min: int | None = None
    year_max: int | None = None
    center: str | None = None
    subject_category: str | None = None
    doc_ids: list[str] = Field(default_factory=list)
    rewritten_query: str | None = None


class RetrievedChunk(BaseModel):
    """A retrieved chunk with stage-by-stage scoring for full observability."""

    chunk_id: str
    parent_id: str
    doc_id: str
    title: str
    section_title: str
    text: str
    parent_text: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    dense_score: float | None = None
    sparse_score: float | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Citation(BaseModel):
    """Structured citation linking a claim back to a specific NASA document section."""

    citation_index: int
    doc_id: str
    title: str
    section_title: str
    page_start: int | None = None
    ntrs_url: str = ""
    score: float | None = None


class StageTimingsMs(BaseModel):
    """Per-stage latency telemetry in milliseconds."""

    query_analysis_ms: float = 0.0
    retrieval_ms: float = 0.0
    rerank_ms: float = 0.0
    generation_ms: float = 0.0
    total_ms: float = 0.0


class RAGResponse(BaseModel):
    """Full structured response returned by the RAG pipeline and FastAPI endpoint."""

    query: str
    answer: str
    abstained: bool = False
    abstention_reason: str | None = None
    citations: list[Citation] = Field(default_factory=list)
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)
    applied_filters: QueryFilters = Field(default_factory=QueryFilters)
    mode: Literal["dense", "hybrid", "hybrid_rerank"] = "hybrid_rerank"
    timings_ms: StageTimingsMs = Field(default_factory=StageTimingsMs)


class EvalSample(BaseModel):
    """One evaluation test case for retrieval and generation benchmarking."""

    sample_id: str
    question: str
    expected_doc_ids: list[str] = Field(default_factory=list)
    reference_answer: str = ""
    category: Literal["factual", "multi_doc", "unanswerable"] = "factual"
    notes: str = ""

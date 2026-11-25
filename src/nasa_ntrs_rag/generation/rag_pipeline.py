"""Grounded RAG orchestration with strict inline citations, abstention gating, and latency telemetry."""

from __future__ import annotations

import time
from typing import Literal

from nasa_ntrs_rag.config import Settings, get_settings
from nasa_ntrs_rag.generation.llm import BaseLLMClient, create_llm_client
from nasa_ntrs_rag.retrieval.hybrid_store import HybridRetriever
from nasa_ntrs_rag.retrieval.query_router import extract_query_filters
from nasa_ntrs_rag.schemas import (
    Citation,
    RAGResponse,
    RetrievedChunk,
    StageTimingsMs,
)

SYSTEM_PROMPT = """You are an aerospace safety and engineering research assistant grounded
strictly in NASA NTRS technical reports.

Rules you MUST follow:
1. Answer ONLY using facts explicitly supported by the provided Sources below.
2. Cite every factual claim inline using square brackets matching the source number, e.g. [1] or [1][3].
3. Preserve numerical values, units, aircraft models, and safety findings accurately.
4. If the provided sources do NOT contain enough information to answer reliably, explicitly state:
   "Insufficient evidence in the indexed NASA NTRS corpus to answer this question."
   Do NOT speculate or invent facts.
5. Respond in the same language as the user's question."""


def build_context_block(chunks: list[RetrievedChunk]) -> str:
    """Format retrieved parent/child passages with numbered metadata headers for citation."""
    blocks: list[str] = []
    for idx, ch in enumerate(chunks, start=1):
        body = ch.parent_text or ch.text
        page_label = f"p. {ch.page_start}" if ch.page_start else "p. N/A"
        year = ch.metadata.get("publication_year", "N/A")
        blocks.append(
            f"[{idx}] Document: {ch.title} (NASA NTRS {ch.doc_id}, Year: {year}, "
            f"Section: {ch.section_title}, {page_label})\n{body}"
        )
    return "\n\n---\n\n".join(blocks)


def build_citations(chunks: list[RetrievedChunk]) -> list[Citation]:
    """Convert retrieved chunks into structured Citation objects."""
    citations: list[Citation] = []
    for idx, ch in enumerate(chunks, start=1):
        score = (
            ch.rerank_score
            if ch.rerank_score is not None
            else (ch.rrf_score if ch.rrf_score is not None else ch.dense_score)
        )
        citations.append(
            Citation(
                citation_index=idx,
                doc_id=ch.doc_id,
                title=ch.title,
                section_title=ch.section_title,
                page_start=ch.page_start,
                ntrs_url=str(ch.metadata.get("ntrs_url", f"https://ntrs.nasa.gov/citations/{ch.doc_id}")),
                score=round(score, 4) if score is not None else None,
            )
        )
    return citations


class NASANTRSRAGPipeline:
    """Full end-to-end RAG pipeline with observability and calibrated abstention."""

    def __init__(
        self,
        settings: Settings | None = None,
        retriever: HybridRetriever | None = None,
        llm: BaseLLMClient | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.retriever = retriever or HybridRetriever(self.settings)
        self._llm = llm

    @property
    def llm(self) -> BaseLLMClient:
        if self._llm is None:
            self._llm = create_llm_client(self.settings)
        return self._llm

    def ask(
        self,
        query: str,
        mode: Literal["dense", "hybrid", "hybrid_rerank"] = "hybrid_rerank",
        top_k: int | None = None,
        use_metadata_filters: bool = True,
    ) -> RAGResponse:
        """Execute query analysis -> hybrid retrieval -> reranking -> grounded generation."""
        t_total = time.perf_counter()

        t0 = time.perf_counter()
        filters = extract_query_filters(query) if use_metadata_filters else extract_query_filters("")
        query_analysis_ms = (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        chunks = self.retriever.retrieve(
            query=query,
            mode=mode,
            filters=filters,
            top_k=top_k,
            expand_to_parent=True,
        )
        retrieval_and_rerank_ms = (time.perf_counter() - t1) * 1000.0

        # Calibrated abstention gate before LLM call
        if not chunks:
            return RAGResponse(
                query=query,
                answer="Insufficient evidence in the indexed NASA NTRS corpus to answer this question.",
                abstained=True,
                abstention_reason="No chunks matched the query and extracted metadata filters.",
                applied_filters=filters,
                mode=mode,
                timings_ms=StageTimingsMs(
                    query_analysis_ms=round(query_analysis_ms, 2),
                    retrieval_ms=round(retrieval_and_rerank_ms, 2),
                    total_ms=round((time.perf_counter() - t_total) * 1000.0, 2),
                ),
            )

        if mode == "hybrid_rerank":
            top_score = chunks[0].rerank_score or 0.0
            if top_score < self.settings.abstention_score_threshold:
                return RAGResponse(
                    query=query,
                    answer=(
                        "Insufficient evidence in the indexed NASA NTRS corpus to answer this "
                        f"question reliably (top relevance score {top_score:.3f} < "
                        f"threshold {self.settings.abstention_score_threshold:.2f})."
                    ),
                    abstained=True,
                    abstention_reason=f"Top reranker score ({top_score:.3f}) below abstention threshold.",
                    citations=build_citations(chunks),
                    retrieved_chunks=chunks,
                    applied_filters=filters,
                    mode=mode,
                    timings_ms=StageTimingsMs(
                        query_analysis_ms=round(query_analysis_ms, 2),
                        retrieval_ms=round(retrieval_and_rerank_ms, 2),
                        total_ms=round((time.perf_counter() - t_total) * 1000.0, 2),
                    ),
                )

        context_block = build_context_block(chunks)
        user_prompt = f"Sources:\n{context_block}\n\nQuestion:\n{query}\n\nGrounded Answer:"

        t2 = time.perf_counter()
        answer = self.llm.generate(system_prompt=SYSTEM_PROMPT, user_prompt=user_prompt)
        generation_ms = (time.perf_counter() - t2) * 1000.0

        abstained = "insufficient evidence" in answer.lower()
        abstention_msg = (
            "Model abstained due to insufficient supporting evidence." if abstained else None
        )
        return RAGResponse(
            query=query,
            answer=answer,
            abstained=abstained,
            abstention_reason=abstention_msg,
            citations=build_citations(chunks),
            retrieved_chunks=chunks,
            applied_filters=filters,
            mode=mode,
            timings_ms=StageTimingsMs(
                query_analysis_ms=round(query_analysis_ms, 2),
                retrieval_ms=round(retrieval_and_rerank_ms, 2),
                generation_ms=round(generation_ms, 2),
                total_ms=round((time.perf_counter() - t_total) * 1000.0, 2),
            ),
        )

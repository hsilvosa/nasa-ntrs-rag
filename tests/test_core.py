"""Unit tests for chunking, hybrid RRF fusion, query filtering, IR metrics, and abstention."""

from __future__ import annotations

from nasa_ntrs_rag.evaluation.metrics import mrr_at_k, ndcg_at_k, recall_at_k
from nasa_ntrs_rag.generation.rag_pipeline import NASANTRSRAGPipeline
from nasa_ntrs_rag.ingestion.chunker import fixed_chunk_document, hierarchical_chunk_document
from nasa_ntrs_rag.retrieval.hybrid_store import reciprocal_rank_fusion, tokenize_technical_text
from nasa_ntrs_rag.retrieval.query_router import extract_query_filters
from nasa_ntrs_rag.schemas import NTRSDocumentMetadata, QueryFilters, RetrievedChunk


def _sample_doc() -> NTRSDocumentMetadata:
    return NTRSDocumentMetadata(
        doc_id="20210012345",
        title="Aircraft Icing Effects on Longitudinal Stability",
        publication_year=2021,
        center="Glenn Research Center",
        ntrs_url="https://ntrs.nasa.gov/citations/20210012345",
    )


def test_hierarchical_chunker_preserves_sections_pages_and_tables() -> None:
    md = (
        "<!-- page: 1 -->\n"
        "## 1. Introduction\n"
        "Supercooled large droplets cause rapid ice accretion on wing leading edges.\n\n"
        "<!-- page: 2 -->\n"
        "## 2. Wind Tunnel Results\n"
        "| AoA (deg) | Clean Cl | Iced Cl |\n"
        "| --- | --- | --- |\n"
        "| 4.0 | 0.52 | 0.48 |\n"
        "| 12.0 | 1.35 | 0.89 |\n\n"
        "Lift degradation exceeded 34 percent near stall."
    )
    parents, children = hierarchical_chunk_document(
        doc=_sample_doc(),
        markdown_text=md,
        parent_chars=2000,
        child_chars=300,
        child_overlap=50,
    )
    assert len(parents) == 2
    assert parents[0].section_title == "1. Introduction"
    assert parents[0].page_start == 1
    assert parents[1].section_title == "2. Wind Tunnel Results"
    assert parents[1].page_start == 2
    assert any("| 12.0 | 1.35 | 0.89 |" in c.text for c in children)
    assert all("NASA NTRS 20210012345" in c.contextual_text for c in children)


def test_fixed_baseline_chunker() -> None:
    text = "<!-- page: 1 -->\n" + ("Flight control law verification. " * 60)
    parents, children = fixed_chunk_document(_sample_doc(), text, chunk_chars=400, overlap=50)
    assert len(parents) == len(children) >= 3
    assert children[0].section_title == "Unstructured"


def test_tokenize_and_rrf_fusion() -> None:
    tokens = tokenize_technical_text("Failure in NASA-TM-20210012345 on B737-MAX Pitot probe")
    assert "nasa-tm-20210012345" in tokens
    assert "b737-max" in tokens

    fused = reciprocal_rank_fusion(
        dense_ranks={"a": 1, "b": 2, "c": 3},
        sparse_ranks={"b": 1, "c": 2, "d": 3},
        rrf_k=60,
    )
    assert fused["b"] > fused["a"]
    assert fused["b"] > fused["d"]


def test_query_filter_extraction() -> None:
    f1 = extract_query_filters(
        "Compare Langley icing wind tunnel tests between 2012 and 2022 referencing 20210012345"
    )
    assert f1.year_min == 2012
    assert f1.year_max == 2022
    assert f1.center == "Langley Research Center"
    assert f1.doc_ids == ["20210012345"]

    f2 = extract_query_filters("Stall recovery procedures since 2018")
    assert f2.year_min == 2018
    assert f2.year_max is None


def test_ir_metrics() -> None:
    retrieved = ["d9", "d1", "d2", "d3"]
    expected = ["d1", "d4"]
    assert recall_at_k(retrieved, expected, k=5) == 0.5
    assert mrr_at_k(retrieved, expected, k=10) == 0.5
    assert 0.0 < ndcg_at_k(retrieved, expected, k=10) < 1.0


class _StubRetriever:
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self._chunks = chunks

    def retrieve(
        self,
        query: str,
        mode: str = "hybrid_rerank",
        filters: QueryFilters | None = None,
        top_k: int | None = None,
        expand_to_parent: bool = True,
    ) -> list[RetrievedChunk]:
        return self._chunks


class _StubLLM:
    def __init__(self) -> None:
        self.called = False

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.called = True
        return "Ice accretion reduced maximum lift by 34% [1]."


def test_abstention_gate_blocks_low_score_without_calling_llm() -> None:
    low_chunk = RetrievedChunk(
        chunk_id="c1",
        parent_id="p1",
        doc_id="20210012345",
        title="Icing Report",
        section_title="Results",
        text="Irrelevant snippet",
        rerank_score=0.04,
    )
    stub_llm = _StubLLM()
    pipeline = NASANTRSRAGPipeline(retriever=_StubRetriever([low_chunk]), llm=stub_llm)  # type: ignore[arg-type]
    resp = pipeline.ask("What is the best paella recipe?", mode="hybrid_rerank")
    assert resp.abstained is True
    assert stub_llm.called is False


def test_grounded_generation_returns_citations() -> None:
    good_chunk = RetrievedChunk(
        chunk_id="c1",
        parent_id="p1",
        doc_id="20210012345",
        title="Icing Report",
        section_title="Results",
        text="Lift dropped 34%.",
        parent_text="Full section: Lift dropped 34% due to leading-edge ice.",
        page_start=2,
        rerank_score=0.91,
        metadata={"publication_year": 2021, "ntrs_url": "https://ntrs.nasa.gov/citations/20210012345"},
    )
    stub_llm = _StubLLM()
    pipeline = NASANTRSRAGPipeline(retriever=_StubRetriever([good_chunk]), llm=stub_llm)  # type: ignore[arg-type]
    resp = pipeline.ask("How much did lift drop?", mode="hybrid_rerank")
    assert resp.abstained is False
    assert stub_llm.called is True
    assert len(resp.citations) == 1
    assert resp.citations[0].doc_id == "20210012345"

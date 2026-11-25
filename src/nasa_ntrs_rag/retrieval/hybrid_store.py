"""Hybrid retrieval combining Qdrant dense vectors, BM25 lexical search, RRF, and Parent-Child expansion."""

from __future__ import annotations

import logging
import re
import uuid
from typing import TYPE_CHECKING, Literal

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    PointStruct,
    Range,
    VectorParams,
)
from rank_bm25 import BM25Okapi

from nasa_ntrs_rag.config import Settings, get_settings
from nasa_ntrs_rag.ingestion.pipeline import load_chunks
from nasa_ntrs_rag.schemas import ChildChunk, ParentChunk, QueryFilters, RetrievedChunk

if TYPE_CHECKING:
    from nasa_ntrs_rag.retrieval.embeddings import CrossEncoderReranker, DenseEmbedder

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[A-Za-z0-9\-_.]+")


def tokenize_technical_text(text: str) -> list[str]:
    """Tokenize aerospace/technical text preserving codes like NASA-TM-20210012345 or B737."""
    return [tok.lower() for tok in _TOKEN_RE.findall(text)]


def reciprocal_rank_fusion(
    dense_ranks: dict[str, int],
    sparse_ranks: dict[str, int],
    rrf_k: int = 60,
) -> dict[str, float]:
    """Fuse dense and sparse rankings via Reciprocal Rank Fusion (Cormack et al., 2009)."""
    all_ids = set(dense_ranks) | set(sparse_ranks)
    fused: dict[str, float] = {}
    for cid in all_ids:
        score = 0.0
        if cid in dense_ranks:
            score += 1.0 / (rrf_k + dense_ranks[cid])
        if cid in sparse_ranks:
            score += 1.0 / (rrf_k + sparse_ranks[cid])
        fused[cid] = score
    return fused


def _chunk_matches_filters(chunk: ChildChunk, filters: QueryFilters) -> bool:
    """Apply the same metadata filters to BM25 candidates as Qdrant applies to dense search."""
    year = chunk.metadata.get("publication_year")
    if filters.year_min is not None and (year is None or year < filters.year_min):
        return False
    if filters.year_max is not None and (year is None or year > filters.year_max):
        return False
    if filters.center:
        center_val = str(chunk.metadata.get("center") or "").lower()
        if filters.center.lower() not in center_val:
            return False
    if filters.doc_ids and chunk.doc_id not in filters.doc_ids:
        return False
    return True


def _build_qdrant_filter(filters: QueryFilters) -> Filter | None:
    """Translate QueryFilters into a Qdrant Filter object."""
    conditions = []
    if filters.year_min is not None or filters.year_max is not None:
        conditions.append(
            FieldCondition(
                key="publication_year",
                range=Range(gte=filters.year_min, lte=filters.year_max),
            )
        )
    if filters.center:
        conditions.append(
            FieldCondition(key="center", match=MatchValue(value=filters.center))
        )
    if filters.doc_ids:
        conditions.append(
            FieldCondition(key="doc_id", match=MatchAny(any=filters.doc_ids))
        )
    return Filter(must=conditions) if conditions else None


class HybridRetriever:
    """End-to-end retrieval engine supporting Dense, Hybrid (RRF), and Hybrid + Rerank modes."""

    def __init__(
        self,
        settings: Settings | None = None,
        embedder: DenseEmbedder | None = None,
        reranker: CrossEncoderReranker | None = None,
        collection_name: str | None = None,
        chunks_prefix: str = "",
    ) -> None:
        self.settings = settings or get_settings()
        self.collection_name = collection_name or self.settings.qdrant_collection
        self.chunks_prefix = chunks_prefix
        self._embedder = embedder
        self._reranker = reranker

        if self.settings.qdrant_url:
            self.client = QdrantClient(url=self.settings.qdrant_url)
        else:
            self.settings.qdrant_local_path.mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=str(self.settings.qdrant_local_path))

        self.parents: dict[str, ParentChunk] = {}
        self.children: list[ChildChunk] = []
        self.children_by_id: dict[str, ChildChunk] = {}
        self._bm25: BM25Okapi | None = None
        self._load_local_chunks()

    @property
    def embedder(self) -> DenseEmbedder:
        if self._embedder is None:
            from nasa_ntrs_rag.retrieval.embeddings import DenseEmbedder

            self._embedder = DenseEmbedder(
                model_name=self.settings.embedding_model,
                device=self.settings.resolve_device(),
                batch_size=self.settings.embedding_batch_size,
            )
        return self._embedder

    @property
    def reranker(self) -> CrossEncoderReranker:
        if self._reranker is None:
            from nasa_ntrs_rag.retrieval.embeddings import CrossEncoderReranker

            self._reranker = CrossEncoderReranker(
                model_name=self.settings.reranker_model,
                device=self.settings.resolve_device(),
            )
        return self._reranker

    def _load_local_chunks(self) -> None:
        self.parents, self.children = load_chunks(
            self.settings.processed_chunks_dir, prefix=self.chunks_prefix
        )
        self.children_by_id = {c.chunk_id: c for c in self.children}
        if self.children:
            corpus_tokens = [tokenize_technical_text(c.contextual_text) for c in self.children]
            self._bm25 = BM25Okapi(corpus_tokens)

    def index_chunks(
        self,
        parents: list[ParentChunk] | None = None,
        children: list[ChildChunk] | None = None,
        recreate: bool = True,
    ) -> int:
        """Embed child chunks and upsert them into Qdrant alongside BM25 refresh."""
        if parents is not None and children is not None:
            self.parents = {p.parent_id: p for p in parents}
            self.children = list(children)
            self.children_by_id = {c.chunk_id: c for c in self.children}
            corpus_tokens = [tokenize_technical_text(c.contextual_text) for c in self.children]
            self._bm25 = BM25Okapi(corpus_tokens) if corpus_tokens else None
        else:
            self._load_local_chunks()

        if not self.children:
            raise ValueError("No child chunks available to index.")

        vectors = self.embedder.encode([c.contextual_text for c in self.children], is_query=False)
        dim = vectors.shape[1]

        exists = self.client.collection_exists(self.collection_name)
        if exists and recreate:
            self.client.delete_collection(self.collection_name)
            exists = False
        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )

        points: list[PointStruct] = []
        for chunk, vec in zip(self.children, vectors, strict=True):
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, chunk.chunk_id))
            payload = {
                "chunk_id": chunk.chunk_id,
                "parent_id": chunk.parent_id,
                "doc_id": chunk.doc_id,
                "title": chunk.title,
                "section_title": chunk.section_title,
                "text": chunk.text,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                **chunk.metadata,
            }
            points.append(PointStruct(id=point_id, vector=vec.tolist(), payload=payload))

        batch_size = 128
        for start in range(0, len(points), batch_size):
            self.client.upsert(
                collection_name=self.collection_name,
                points=points[start : start + batch_size],
            )
        logger.info("Indexed %d child chunks into Qdrant collection '%s'", len(points), self.collection_name)
        return len(points)

    def _search_dense(
        self, query: str, top_k: int, filters: QueryFilters
    ) -> list[tuple[str, float]]:
        query_vec = self.embedder.encode([query], is_query=True)[0].tolist()
        qdrant_filter = _build_qdrant_filter(filters)
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vec,
            query_filter=qdrant_filter,
            limit=top_k,
            with_payload=True,
        )
        return [
            (str(pt.payload["chunk_id"]), float(pt.score))
            for pt in response.points
            if pt.payload and "chunk_id" in pt.payload
        ]

    def _search_sparse(
        self, query: str, top_k: int, filters: QueryFilters
    ) -> list[tuple[str, float]]:
        if self._bm25 is None or not self.children:
            return []
        tokens = tokenize_technical_text(query)
        if not tokens:
            return []
        raw_scores = self._bm25.get_scores(tokens)
        scored: list[tuple[str, float]] = []
        for chunk, score in zip(self.children, raw_scores, strict=True):
            if score <= 0:
                continue
            if not _chunk_matches_filters(chunk, filters):
                continue
            scored.append((chunk.chunk_id, float(score)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def retrieve(
        self,
        query: str,
        mode: Literal["dense", "hybrid", "hybrid_rerank"] = "hybrid_rerank",
        filters: QueryFilters | None = None,
        top_k: int | None = None,
        expand_to_parent: bool = True,
    ) -> list[RetrievedChunk]:
        """Run dense, hybrid (RRF), or hybrid + cross-encoder reranking retrieval."""
        filters = filters or QueryFilters()
        final_k = top_k or self.settings.retrieval_top_k_rerank
        search_query = filters.rewritten_query or query

        dense_hits = self._search_dense(
            search_query,
            top_k=self.settings.retrieval_top_k_dense if mode != "dense" else final_k,
            filters=filters,
        )
        dense_score_map = dict(dense_hits)
        dense_rank_map = {cid: rank for rank, (cid, _) in enumerate(dense_hits, start=1)}

        sparse_score_map: dict[str, float] = {}
        sparse_rank_map: dict[str, int] = {}
        if mode in ("hybrid", "hybrid_rerank"):
            sparse_hits = self._search_sparse(
                search_query,
                top_k=self.settings.retrieval_top_k_sparse,
                filters=filters,
            )
            sparse_score_map = dict(sparse_hits)
            sparse_rank_map = {cid: rank for rank, (cid, _) in enumerate(sparse_hits, start=1)}
            fused = reciprocal_rank_fusion(
                dense_rank_map, sparse_rank_map, rrf_k=self.settings.rrf_k
            )
            ordered_ids = [
                cid for cid, _ in sorted(fused.items(), key=lambda x: x[1], reverse=True)
            ]
        else:
            fused = {}
            ordered_ids = [cid for cid, _ in dense_hits]

        candidate_ids = ordered_ids[: self.settings.retrieval_top_k_dense]
        candidates: list[RetrievedChunk] = []
        for cid in candidate_ids:
            child = self.children_by_id.get(cid)
            if child is None:
                continue
            parent = self.parents.get(child.parent_id)
            candidates.append(
                RetrievedChunk(
                    chunk_id=child.chunk_id,
                    parent_id=child.parent_id,
                    doc_id=child.doc_id,
                    title=child.title,
                    section_title=child.section_title,
                    text=child.text,
                    parent_text=parent.text if (expand_to_parent and parent) else None,
                    page_start=child.page_start,
                    page_end=child.page_end,
                    dense_score=dense_score_map.get(cid),
                    sparse_score=sparse_score_map.get(cid),
                    rrf_score=fused.get(cid),
                    metadata=child.metadata,
                )
            )

        if mode == "hybrid_rerank" and candidates:
            passages = [
                self.children_by_id[c.chunk_id].contextual_text
                if c.chunk_id in self.children_by_id
                else c.text
                for c in candidates
            ]
            scores = self.reranker.score(query, passages)
            for cand, sc in zip(candidates, scores, strict=True):
                cand.rerank_score = float(sc)
            candidates.sort(key=lambda c: c.rerank_score or 0.0, reverse=True)

        # Deduplicate by parent_id while keeping the highest-ranked child per section
        if expand_to_parent:
            deduped: list[RetrievedChunk] = []
            seen_parents: set[str] = set()
            for cand in candidates:
                if cand.parent_id in seen_parents:
                    continue
                seen_parents.add(cand.parent_id)
                deduped.append(cand)
                if len(deduped) >= final_k:
                    break
            return deduped

        return candidates[:final_k]

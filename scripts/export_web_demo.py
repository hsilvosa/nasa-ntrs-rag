"""Export real GPU RAG traces and a compact 2,426-report searchable JSON index for the web demo."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Literal

from nasa_ntrs_rag.config import get_settings
from nasa_ntrs_rag.generation.rag_pipeline import NASANTRSRAGPipeline
from nasa_ntrs_rag.ingestion.ntrs_client import load_manifest
from nasa_ntrs_rag.retrieval.hybrid_store import HybridRetriever
from nasa_ntrs_rag.retrieval.query_router import extract_query_filters

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

DEMO_QUERIES: list[dict[str, str]] = [
    {
        "id": "columbia-foam",
        "label": "Columbia CAIB foam impact",
        "category": "Accident investigation",
        "question": "What physical and organizational causes did the Columbia Accident Investigation Board identify for the loss of Space Shuttle Columbia?",
    },
    {
        "id": "turbofan-icing",
        "label": "Turbofan ice crystal icing",
        "category": "Aircraft icing",
        "question": "How does ice crystal ingestion affect commercial high-bypass turbofan engines and what atmospheric conditions create icing risk?",
    },
    {
        "id": "fault-tolerant-fcs",
        "label": "Fault-tolerant flight control",
        "category": "Flight control",
        "question": "How do restructurable and rule-based fault-tolerant flight control systems handle actuator or control surface failures?",
    },
    {
        "id": "single-pilot-workload",
        "label": "Single-pilot workload",
        "category": "Human factors",
        "question": "What cognitive workload challenges do integrated glass cockpits create for single-pilot operations and how can automation help?",
    },
    {
        "id": "runway-safety",
        "label": "RUNSAFE runway safety",
        "category": "Runway safety",
        "question": "What landing and runway excursion hazards were investigated in NASA runway safety and RUNSAFE evaluations?",
    },
    {
        "id": "upset-recovery",
        "label": "Stall & upset recovery",
        "category": "Aerodynamic stall",
        "question": "What pilot training gaps and flight physics issues affect aircraft upset prevention and aerodynamic stall recovery?",
    },
    {
        "id": "avionics-verification",
        "label": "Avionics formal verification",
        "category": "Avionics software",
        "question": "How is formal verification or floating-point round-off error analysis applied to safety-critical avionics software and microprocessors?",
    },
    {
        "id": "thermal-protection",
        "label": "Thermal protection reentry",
        "category": "Thermal protection",
        "question": "What materials, aerothermal loads, and failure modes govern spacecraft thermal protection systems during atmospheric reentry?",
    },
    {
        "id": "filtered-glenn",
        "label": "Filtered: Glenn icing (2010–2024)",
        "category": "Metadata filter routing",
        "question": "What aircraft icing findings were published by Glenn Research Center between 2010 and 2024?",
    },
    {
        "id": "ood-paella",
        "label": "Out-of-domain: Paella recipe",
        "category": "Abstention gate",
        "question": "Which Mediterranean recipe uses saffron and smoked paprika for seafood paella?",
    },
    {
        "id": "ood-stock",
        "label": "Out-of-domain: Boeing stock 2026",
        "category": "Abstention gate",
        "question": "What was the exact stock closing price of Boeing on March 14, 2026?",
    },
    {
        "id": "ood-ibuprofen",
        "label": "Out-of-domain: Pediatric dosing",
        "category": "Abstention gate",
        "question": "What is the recommended dosage of ibuprofen for pediatric patients under 12 kg?",
    },
]

MODES: tuple[Literal["dense", "hybrid", "hybrid_rerank"], ...] = (
    "dense",
    "hybrid",
    "hybrid_rerank",
)


def _trim(text: str, limit: int) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1].rstrip() + "…"


def export_corpus_index(out_path: Path) -> dict[str, Any]:
    """Export a compact searchable JSON index of all 2,426 downloaded NASA NTRS reports."""
    settings = get_settings()
    docs = load_manifest(settings.manifest_path)

    records: list[dict[str, Any]] = []
    centers: dict[str, int] = {}
    topics: dict[str, int] = {}

    for doc in docs:
        center = doc.center or "NASA"
        centers[center] = centers.get(center, 0) + 1
        topic = doc.topic_query or "general"
        topics[topic] = topics.get(topic, 0) + 1

        records.append(
            {
                "id": doc.doc_id,
                "t": _trim(doc.title, 180),
                "a": _trim(doc.abstract, 320) if doc.abstract else "",
                "y": doc.publication_year,
                "c": center,
                "q": topic,
                "s": doc.sti_type or "",
                "u": doc.ntrs_url or f"https://ntrs.nasa.gov/citations/{doc.doc_id}",
            }
        )

    payload = {
        "generated_at": time.strftime("%Y-%m-%d"),
        "doc_count": len(records),
        "top_centers": sorted(centers.items(), key=lambda x: x[1], reverse=True)[:12],
        "top_topics": sorted(topics.items(), key=lambda x: x[1], reverse=True)[:20],
        "docs": records,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    logger.info("Wrote corpus index (%d docs, %.2f MB) to %s", len(records), out_path.stat().st_size / 1e6, out_path)
    return payload


def export_traces(out_path: Path, with_llm: bool = True) -> dict[str, Any]:
    """Run all demo queries across dense, hybrid, and hybrid_rerank and save real GPU traces."""
    settings = get_settings()
    retriever = HybridRetriever(settings=settings)
    pipeline = NASANTRSRAGPipeline(settings=settings, retriever=retriever)

    ablation_path = settings.eval_dir / "ablation_results.json"
    ablation = json.loads(ablation_path.read_text(encoding="utf-8")) if ablation_path.exists() else {}

    exported_queries: list[dict[str, Any]] = []

    for item in DEMO_QUERIES:
        q = item["question"]
        filters = extract_query_filters(q)
        logger.info("Running trace '%s': %s", item["id"], q[:72])
        mode_results: dict[str, Any] = {}

        for mode in MODES:
            if with_llm:
                resp = pipeline.ask(query=q, mode=mode, top_k=5, use_metadata_filters=True)
                chunks_out = []
                for idx, ch in enumerate(resp.retrieved_chunks, start=1):
                    chunks_out.append(
                        {
                            "citation_index": idx,
                            "chunk_id": ch.chunk_id,
                            "doc_id": ch.doc_id,
                            "title": _trim(ch.title, 160),
                            "section_title": _trim(ch.section_title, 90),
                            "page_start": ch.page_start,
                            "year": ch.metadata.get("publication_year"),
                            "center": ch.metadata.get("center"),
                            "ntrs_url": str(
                                ch.metadata.get("ntrs_url", f"https://ntrs.nasa.gov/citations/{ch.doc_id}")
                            ),
                            "dense_score": round(ch.dense_score, 4) if ch.dense_score is not None else None,
                            "sparse_score": round(ch.sparse_score, 2) if ch.sparse_score is not None else None,
                            "rrf_score": round(ch.rrf_score, 5) if ch.rrf_score is not None else None,
                            "rerank_score": round(ch.rerank_score, 4) if ch.rerank_score is not None else None,
                            "text": _trim(ch.text, 560),
                            "parent_excerpt": _trim(ch.parent_text or ch.text, 1100),
                        }
                    )
                mode_results[mode] = {
                    "answer": resp.answer,
                    "abstained": resp.abstained,
                    "abstention_reason": resp.abstention_reason,
                    "timings_ms": resp.timings_ms.model_dump(),
                    "citations": [c.model_dump() for c in resp.citations],
                    "chunks": chunks_out,
                }
            else:
                t0 = time.perf_counter()
                chunks = retriever.retrieve(query=q, mode=mode, filters=filters, top_k=5, expand_to_parent=True)
                retrieval_ms = round((time.perf_counter() - t0) * 1000.0, 1)
                top_rerank = chunks[0].rerank_score if (mode == "hybrid_rerank" and chunks) else None
                abstained = bool(
                    (not chunks)
                    or (
                        mode == "hybrid_rerank"
                        and (top_rerank or 0.0) < settings.abstention_score_threshold
                    )
                )
                chunks_out = []
                for idx, ch in enumerate(chunks, start=1):
                    chunks_out.append(
                        {
                            "citation_index": idx,
                            "chunk_id": ch.chunk_id,
                            "doc_id": ch.doc_id,
                            "title": _trim(ch.title, 160),
                            "section_title": _trim(ch.section_title, 90),
                            "page_start": ch.page_start,
                            "year": ch.metadata.get("publication_year"),
                            "center": ch.metadata.get("center"),
                            "ntrs_url": str(
                                ch.metadata.get("ntrs_url", f"https://ntrs.nasa.gov/citations/{ch.doc_id}")
                            ),
                            "dense_score": round(ch.dense_score, 4) if ch.dense_score is not None else None,
                            "sparse_score": round(ch.sparse_score, 2) if ch.sparse_score is not None else None,
                            "rrf_score": round(ch.rrf_score, 5) if ch.rrf_score is not None else None,
                            "rerank_score": round(ch.rerank_score, 4) if ch.rerank_score is not None else None,
                            "text": _trim(ch.text, 560),
                            "parent_excerpt": _trim(ch.parent_text or ch.text, 1100),
                        }
                    )
                mode_results[mode] = {
                    "answer": (
                        f"Insufficient evidence in the indexed NASA NTRS corpus to answer this question reliably "
                        f"(top reranker score {(top_rerank or 0.0):.3f} < threshold {settings.abstention_score_threshold:.2f})."
                        if abstained
                        else "Retrieval-only trace (LLM generation disabled)."
                    ),
                    "abstained": abstained,
                    "abstention_reason": (
                        f"Top reranker score ({(top_rerank or 0.0):.3f}) below abstention threshold."
                        if abstained
                        else None
                    ),
                    "timings_ms": {"retrieval_ms": retrieval_ms, "generation_ms": 0.0, "total_ms": retrieval_ms},
                    "citations": [],
                    "chunks": chunks_out,
                }

        exported_queries.append(
            {
                "id": item["id"],
                "label": item["label"],
                "category": item["category"],
                "question": q,
                "applied_filters": filters.model_dump(),
                "modes": mode_results,
            }
        )

    payload = {
        "generated_at": time.strftime("%Y-%m-%d"),
        "hardware": "NVIDIA RTX 3060 12 GB (CUDA fp16)",
        "embedding_model": settings.embedding_model,
        "reranker_model": settings.reranker_model,
        "llm_model": settings.llm_model if with_llm else None,
        "abstention_threshold": settings.abstention_score_threshold,
        "corpus": {
            "documents": len(load_manifest(settings.manifest_path)),
            "parent_chunks": len(retriever.parents),
            "child_chunks": len(retriever.children),
            "size_gb": 10.01,
        },
        "ablation": ablation,
        "queries": exported_queries,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Wrote %d GPU traces (%.1f KB) to %s", len(exported_queries), out_path.stat().st_size / 1e3, out_path)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("C:/Users/Usuario/Desktop/PORTFOLIO/personal-website/public/data/nasa-ntrs"),
    )
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM answer generation")
    args = parser.parse_args()

    export_corpus_index(args.out_dir / "corpus.json")
    export_traces(args.out_dir / "traces.json", with_llm=not args.no_llm)


if __name__ == "__main__":
    main()

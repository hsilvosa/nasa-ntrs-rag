"""Deterministic IR evaluation metrics (Recall@K, MRR@K, nDCG@K) and ablation runner."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Literal

from nasa_ntrs_rag.ingestion.ntrs_client import load_manifest
from nasa_ntrs_rag.retrieval.hybrid_store import HybridRetriever
from nasa_ntrs_rag.retrieval.query_router import extract_query_filters
from nasa_ntrs_rag.schemas import EvalSample


def recall_at_k(retrieved_doc_ids: list[str], expected_doc_ids: list[str], k: int = 5) -> float:
    """Fraction of expected relevant documents retrieved within top-k unique doc IDs."""
    if not expected_doc_ids:
        return 1.0
    top_k_unique = list(dict.fromkeys(retrieved_doc_ids))[:k]
    hits = sum(1 for doc_id in expected_doc_ids if doc_id in top_k_unique)
    return hits / len(expected_doc_ids)


def mrr_at_k(retrieved_doc_ids: list[str], expected_doc_ids: list[str], k: int = 10) -> float:
    """Reciprocal rank of the first relevant document within top-k."""
    if not expected_doc_ids:
        return 1.0
    expected_set = set(expected_doc_ids)
    for rank, doc_id in enumerate(list(dict.fromkeys(retrieved_doc_ids))[:k], start=1):
        if doc_id in expected_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_doc_ids: list[str], expected_doc_ids: list[str], k: int = 10) -> float:
    """Binary-relevance normalized Discounted Cumulative Gain at k."""
    if not expected_doc_ids:
        return 1.0
    expected_set = set(expected_doc_ids)
    ranked = list(dict.fromkeys(retrieved_doc_ids))[:k]
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, doc_id in enumerate(ranked, start=1)
        if doc_id in expected_set
    )
    ideal_hits = min(len(expected_set), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def load_eval_dataset(path: Path) -> list[EvalSample]:
    samples: list[EvalSample] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                samples.append(EvalSample.model_validate_json(line))
    return samples


def save_eval_dataset(path: Path, samples: list[EvalSample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for s in samples:
            fh.write(s.model_dump_json() + "\n")


def build_starter_eval_from_manifest(
    manifest_path: Path,
    output_path: Path,
    max_factual: int = 30,
) -> list[EvalSample]:
    """Generate a deterministic starter eval set from NTRS abstracts + out-of-domain controls."""
    docs = load_manifest(manifest_path)
    samples: list[EvalSample] = []

    for idx, doc in enumerate(docs[:max_factual], start=1):
        snippet = doc.abstract.split(".")[0].strip() if doc.abstract else doc.title
        question = (
            f"What technical findings or safety issues are analyzed in the NASA report "
            f"regarding '{doc.title}'?"
        )
        samples.append(
            EvalSample(
                sample_id=f"ntrs-f-{idx:03d}",
                question=question,
                expected_doc_ids=[doc.doc_id],
                reference_answer=snippet,
                category="factual",
                notes=f"Auto-seeded from NTRS {doc.doc_id} ({doc.publication_year})",
            )
        )

    unanswerable_questions = [
        "What was the exact stock closing price of Boeing on March 14, 2026?",
        "Which Mediterranean recipe uses saffron and smoked paprika for seafood paella?",
        "What are the GDPR Article 17 fines issued by Spain's AEPD in 2025?",
        "How many UEFA Champions League trophies has Real Madrid won?",
        "What is the recommended dosage of ibuprofen for pediatric patients under 12 kg?",
    ]
    for idx, q in enumerate(unanswerable_questions, start=1):
        samples.append(
            EvalSample(
                sample_id=f"ntrs-u-{idx:03d}",
                question=q,
                expected_doc_ids=[],
                category="unanswerable",
                notes="Out-of-domain abstention control",
            )
        )

    save_eval_dataset(output_path, samples)
    return samples


def run_retrieval_ablation(
    retriever: HybridRetriever,
    samples: list[EvalSample],
    modes: tuple[Literal["dense", "hybrid", "hybrid_rerank"], ...] = (
        "dense",
        "hybrid",
        "hybrid_rerank",
    ),
    top_k: int = 10,
    abstention_threshold: float = 0.15,
    output_json_path: Path | None = None,
) -> dict[str, Any]:
    """Benchmark retrieval modes across Recall@5, MRR@10, nDCG@10, abstention, and latency."""
    report: dict[str, Any] = {"num_samples": len(samples), "modes": {}}

    for mode in modes:
        recalls: list[float] = []
        mrrs: list[float] = []
        ndcgs: list[float] = []
        latencies_ms: list[float] = []
        unanswerable_total = 0
        unanswerable_abstained = 0

        for sample in samples:
            t0 = time.perf_counter()
            filters = extract_query_filters(sample.question)
            chunks = retriever.retrieve(
                query=sample.question,
                mode=mode,
                filters=filters,
                top_k=top_k,
                expand_to_parent=False,
            )
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)
            retrieved_docs = [c.doc_id for c in chunks]

            if sample.category == "unanswerable":
                unanswerable_total += 1
                if mode == "hybrid_rerank":
                    top_score = chunks[0].rerank_score if chunks else 0.0
                    if (top_score or 0.0) < abstention_threshold:
                        unanswerable_abstained += 1
                elif not chunks:
                    unanswerable_abstained += 1
            else:
                recalls.append(recall_at_k(retrieved_docs, sample.expected_doc_ids, k=5))
                mrrs.append(mrr_at_k(retrieved_docs, sample.expected_doc_ids, k=10))
                ndcgs.append(ndcg_at_k(retrieved_docs, sample.expected_doc_ids, k=10))

        latencies_sorted = sorted(latencies_ms)
        p95_idx = min(len(latencies_sorted) - 1, int(0.95 * len(latencies_sorted))) if latencies_sorted else 0

        report["modes"][mode] = {
            "recall_at_5": round(sum(recalls) / len(recalls), 4) if recalls else 0.0,
            "mrr_at_10": round(sum(mrrs) / len(mrrs), 4) if mrrs else 0.0,
            "ndcg_at_10": round(sum(ndcgs) / len(ndcgs), 4) if ndcgs else 0.0,
            "abstention_rate_unanswerable": (
                round(unanswerable_abstained / unanswerable_total, 4)
                if unanswerable_total > 0
                else None
            ),
            "latency_p50_ms": round(latencies_sorted[len(latencies_sorted) // 2], 1)
            if latencies_sorted
            else 0.0,
            "latency_p95_ms": round(latencies_sorted[p95_idx], 1) if latencies_sorted else 0.0,
        }

    if output_json_path is not None:
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        output_json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def format_ablation_markdown(report: dict[str, Any]) -> str:
    """Format an ablation report as a GitHub-Flavored Markdown benchmark table."""
    lines = [
        (
            "| Mode | Recall@5 | MRR@10 | nDCG@10 | Unanswerable Abstention | "
            "Latency p50 (ms) | Latency p95 (ms) |"
        ),
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for mode, stats in report.get("modes", {}).items():
        abst = (
            f"{stats['abstention_rate_unanswerable']:.1%}"
            if stats.get("abstention_rate_unanswerable") is not None
            else "N/A"
        )
        lines.append(
            f"| `{mode}` | {stats['recall_at_5']:.3f} | {stats['mrr_at_10']:.3f} | "
            f"{stats['ndcg_at_10']:.3f} | {abst} | {stats['latency_p50_ms']:.1f} | "
            f"{stats['latency_p95_ms']:.1f} |"
        )
    return "\n".join(lines)

"""Build a license-filtered Parquet release of the NASA NTRS RAG corpus and publish to Hugging Face."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi

from nasa_ntrs_rag.config import get_settings
from nasa_ntrs_rag.ingestion.ntrs_client import load_manifest

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

ALLOWED_COPYRIGHT_TYPES = {
    "GOV_PUBLIC_USE_PERMITTED",
    "PUBLIC_USE_PERMITTED",
    "GOV_PERMITTED",
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_parquet(rows: list[dict[str, Any]], out_path: Path, schema: pa.Schema) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(table, out_path, compression="zstd", compression_level=6)
    return len(rows)


def build_release(out_dir: Path, repo_id: str) -> dict[str, Any]:
    settings = get_settings()
    docs = load_manifest(settings.manifest_path)

    allowed_ids: set[str] = set()
    copyright_by_id: dict[str, str] = {}
    excluded_counts: Counter[str] = Counter()
    kept_counts: Counter[str] = Counter()

    for doc in docs:
        raw_meta_path = settings.raw_metadata_dir / f"{doc.doc_id}.json"
        if not raw_meta_path.exists():
            excluded_counts["MISSING_RAW_METADATA"] += 1
            continue
        raw = json.loads(raw_meta_path.read_text(encoding="utf-8"))
        dist = str(raw.get("distribution") or "")
        ctype = str((raw.get("copyright") or {}).get("determinationType") or "UNKNOWN")
        if dist == "PUBLIC" and ctype in ALLOWED_COPYRIGHT_TYPES:
            allowed_ids.add(doc.doc_id)
            copyright_by_id[doc.doc_id] = ctype
            kept_counts[ctype] += 1
        else:
            excluded_counts[f"{dist}:{ctype}"] += 1

    logger.info(
        "License filter: kept %d / %d docs (excluded %d: %s)",
        len(allowed_ids),
        len(docs),
        len(docs) - len(allowed_ids),
        dict(excluded_counts),
    )

    # 1. Documents table (metadata + full layout-aware Markdown)
    doc_rows: list[dict[str, Any]] = []
    for doc in docs:
        if doc.doc_id not in allowed_ids:
            continue
        md_path = settings.processed_md_dir / f"{doc.doc_id}.md"
        md_text = md_path.read_text(encoding="utf-8") if md_path.exists() else ""
        doc_rows.append(
            {
                "doc_id": doc.doc_id,
                "title": doc.title,
                "abstract": doc.abstract,
                "publication_year": doc.publication_year,
                "center": doc.center,
                "sti_type": doc.sti_type,
                "copyright_determination": copyright_by_id[doc.doc_id],
                "topic_query": doc.topic_query,
                "subject_categories": doc.subject_categories,
                "keywords": doc.keywords,
                "authors": doc.authors,
                "ntrs_url": doc.ntrs_url,
                "markdown_text": md_text,
                "markdown_chars": len(md_text),
            }
        )

    doc_schema = pa.schema(
        [
            ("doc_id", pa.string()),
            ("title", pa.string()),
            ("abstract", pa.string()),
            ("publication_year", pa.int32()),
            ("center", pa.string()),
            ("sti_type", pa.string()),
            ("copyright_determination", pa.string()),
            ("topic_query", pa.string()),
            ("subject_categories", pa.list_(pa.string())),
            ("keywords", pa.list_(pa.string())),
            ("authors", pa.list_(pa.string())),
            ("ntrs_url", pa.string()),
            ("markdown_text", pa.large_string()),
            ("markdown_chars", pa.int32()),
        ]
    )
    docs_pq = out_dir / "data" / "documents" / "train-00000-of-00001.parquet"
    n_docs = _write_parquet(doc_rows, docs_pq, doc_schema)
    logger.info("Wrote documents: %d rows (%.2f MB)", n_docs, docs_pq.stat().st_size / 1e6)

    # 2. Parents table
    parent_rows: list[dict[str, Any]] = []
    parents_jsonl = settings.processed_chunks_dir / "parents.jsonl"
    with parents_jsonl.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj["doc_id"] not in allowed_ids:
                continue
            meta = obj.get("metadata") or {}
            parent_rows.append(
                {
                    "parent_id": obj["parent_id"],
                    "doc_id": obj["doc_id"],
                    "title": obj["title"],
                    "section_title": obj["section_title"],
                    "text": obj["text"],
                    "page_start": obj.get("page_start"),
                    "page_end": obj.get("page_end"),
                    "publication_year": meta.get("publication_year"),
                    "center": meta.get("center"),
                    "sti_type": meta.get("sti_type"),
                    "topic_query": meta.get("topic_query"),
                    "ntrs_url": meta.get("ntrs_url"),
                }
            )

    parent_schema = pa.schema(
        [
            ("parent_id", pa.string()),
            ("doc_id", pa.string()),
            ("title", pa.string()),
            ("section_title", pa.string()),
            ("text", pa.large_string()),
            ("page_start", pa.int32()),
            ("page_end", pa.int32()),
            ("publication_year", pa.int32()),
            ("center", pa.string()),
            ("sti_type", pa.string()),
            ("topic_query", pa.string()),
            ("ntrs_url", pa.string()),
        ]
    )
    parents_pq = out_dir / "data" / "parents" / "train-00000-of-00001.parquet"
    n_parents = _write_parquet(parent_rows, parents_pq, parent_schema)
    logger.info("Wrote parents: %d rows (%.2f MB)", n_parents, parents_pq.stat().st_size / 1e6)
    del parent_rows

    # 3. Children table (write in two balanced shards so every file stays well under 150 MB)
    child_rows: list[dict[str, Any]] = []
    children_jsonl = settings.processed_chunks_dir / "children.jsonl"
    with children_jsonl.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj["doc_id"] not in allowed_ids:
                continue
            meta = obj.get("metadata") or {}
            child_rows.append(
                {
                    "chunk_id": obj["chunk_id"],
                    "parent_id": obj["parent_id"],
                    "doc_id": obj["doc_id"],
                    "chunk_index": obj["chunk_index"],
                    "title": obj["title"],
                    "section_title": obj["section_title"],
                    "text": obj["text"],
                    "contextual_text": obj["contextual_text"],
                    "page_start": obj.get("page_start"),
                    "page_end": obj.get("page_end"),
                    "publication_year": meta.get("publication_year"),
                    "center": meta.get("center"),
                    "sti_type": meta.get("sti_type"),
                    "topic_query": meta.get("topic_query"),
                    "ntrs_url": meta.get("ntrs_url"),
                }
            )

    child_schema = pa.schema(
        [
            ("chunk_id", pa.string()),
            ("parent_id", pa.string()),
            ("doc_id", pa.string()),
            ("chunk_index", pa.int32()),
            ("title", pa.string()),
            ("section_title", pa.string()),
            ("text", pa.large_string()),
            ("contextual_text", pa.large_string()),
            ("page_start", pa.int32()),
            ("page_end", pa.int32()),
            ("publication_year", pa.int32()),
            ("center", pa.string()),
            ("sti_type", pa.string()),
            ("topic_query", pa.string()),
            ("ntrs_url", pa.string()),
        ]
    )
    mid = len(child_rows) // 2
    child_pq_0 = out_dir / "data" / "children" / "train-00000-of-00002.parquet"
    child_pq_1 = out_dir / "data" / "children" / "train-00001-of-00002.parquet"
    n_c0 = _write_parquet(child_rows[:mid], child_pq_0, child_schema)
    n_c1 = _write_parquet(child_rows[mid:], child_pq_1, child_schema)
    n_children = n_c0 + n_c1
    logger.info(
        "Wrote children: %d rows in 2 shards (%.2f MB + %.2f MB)",
        n_children,
        child_pq_0.stat().st_size / 1e6,
        child_pq_1.stat().st_size / 1e6,
    )
    del child_rows

    # 4. Eval table (filtered to verified allowed_ids + out-of-domain controls)
    eval_rows: list[dict[str, Any]] = []
    eval_jsonl = settings.eval_dir / "eval_100.jsonl"
    if eval_jsonl.exists():
        with eval_jsonl.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                exp_ids = obj.get("expected_doc_ids") or []
                if exp_ids and not all(doc_id in allowed_ids for doc_id in exp_ids):
                    continue
                eval_rows.append(
                    {
                        "sample_id": obj["sample_id"],
                        "question": obj["question"],
                        "expected_doc_ids": exp_ids,
                        "reference_answer": obj.get("reference_answer", ""),
                        "category": obj["category"],
                        "notes": obj.get("notes", ""),
                    }
                )

    eval_schema = pa.schema(
        [
            ("sample_id", pa.string()),
            ("question", pa.string()),
            ("expected_doc_ids", pa.list_(pa.string())),
            ("reference_answer", pa.string()),
            ("category", pa.string()),
            ("notes", pa.string()),
        ]
    )
    eval_pq = out_dir / "data" / "eval" / "train-00000-of-00001.parquet"
    n_eval = _write_parquet(eval_rows, eval_pq, eval_schema)
    logger.info("Wrote eval: %d rows (%.3f MB)", n_eval, eval_pq.stat().st_size / 1e6)

    # Checksums
    parquet_files = [docs_pq, parents_pq, child_pq_0, child_pq_1, eval_pq]
    checksums = {str(p.relative_to(out_dir)).replace("\\", "/"): _sha256(p) for p in parquet_files}
    (out_dir / "checksums.json").write_text(json.dumps(checksums, indent=2), encoding="utf-8")

    total_mb = sum(p.stat().st_size for p in parquet_files) / 1e6
    total_rows = n_docs + n_parents + n_children + n_eval

    # Dataset Card (README.md)
    card = f"""---
license: other
license_name: nasa-ntrs-public-use
language:
- en
task_categories:
- text-retrieval
- question-answering
tags:
- nasa
- ntrs
- aerospace
- aviation-safety
- rag
- information-retrieval
- hierarchical-chunking
pretty_name: NASA NTRS Aerospace Safety & Engineering RAG Corpus
configs:
- config_name: documents
  data_files:
  - split: train
    path: data/documents/train-*.parquet
- config_name: parents
  data_files:
  - split: train
    path: data/parents/train-*.parquet
- config_name: children
  data_files:
  - split: train
    path: data/children/train-*.parquet
- config_name: eval
  data_files:
  - split: train
    path: data/eval/train-*.parquet
---

# NASA NTRS Aerospace Safety & Engineering RAG Corpus

A structured, license-filtered Parquet corpus derived from **{n_docs:,} open-access NASA Technical Reports Server (NTRS)** aerospace safety and engineering PDFs, prepared for reproducible hybrid and hierarchical Retrieval-Augmented Generation (RAG) benchmarking.

- **Source code & pipeline:** [github.com/hsilvosa/nasa-ntrs-rag](https://github.com/hsilvosa/nasa-ntrs-rag)
- **Project note & interactive demo:** [hsilvosa.com/projects/nasa-ntrs-rag](https://hsilvosa.com/projects/nasa-ntrs-rag/)
- **Upstream source:** [NASA Technical Reports Server (NTRS)](https://ntrs.nasa.gov/)

## Configurations

| Configuration | Rows | Description |
| :--- | ---: | :--- |
| `documents` | {n_docs:,} | Normalized NTRS metadata + layout-aware structured Markdown (`<!-- page: N -->`, promoted headings, selective GFM tables). |
| `parents` | {n_parents:,} | Section-level parent chunks (~4,800 chars, tables preserved) expanded at generation time for causal context. |
| `children` | {n_children:,} | Fine-grained child chunks (~1,200 chars, 200 overlap) with both raw `text` and `contextual_text` headers (`[NASA NTRS <id> \\| <title> \\| Year \\| Center \\| Section]`). |
| `eval` | {n_eval:,} | Stratified retrieval evaluation queries across all 20 aerospace domains plus out-of-domain unanswerable abstention controls. |

**Total rows across configurations:** {total_rows:,} ({total_mb:.1f} MB ZSTD Parquet; original PDFs are not redistributed).

## Quickstart

```python
from datasets import load_dataset

docs = load_dataset("{repo_id}", "documents", split="train")
parents = load_dataset("{repo_id}", "parents", split="train")
children = load_dataset("{repo_id}", "children", split="train")
eval_set = load_dataset("{repo_id}", "eval", split="train")
```

## License & Copyright Filtering

All records were collected from the public NASA NTRS API (`distribution == "PUBLIC"`). Before exporting text to Parquet, every record was audited against NASA's official `copyright.determinationType` field:

- **Included ({n_docs:,} documents):**
  - `GOV_PUBLIC_USE_PERMITTED`: {kept_counts['GOV_PUBLIC_USE_PERMITTED']:,}
  - `PUBLIC_USE_PERMITTED`: {kept_counts['PUBLIC_USE_PERMITTED']:,}
  - `GOV_PERMITTED`: {kept_counts['GOV_PERMITTED']:,}
- **Excluded ({len(docs) - n_docs:,} documents):** `MAY_INCLUDE_COPYRIGHT_MATERIAL` ({excluded_counts.get('PUBLIC:MAY_INCLUDE_COPYRIGHT_MATERIAL', 0):,}) and `OTHER` ({excluded_counts.get('PUBLIC:OTHER', 0):,}) were excluded from all configurations (`documents`, `parents`, `children`, `eval`) to avoid redistributing third-party copyrighted conference/contractor material.

All original PDFs remain hosted by NASA at their canonical `ntrs_url` (`https://ntrs.nasa.gov/citations/<doc_id>`).

## Domains Covered

Collected in balanced round-robin waves across 20 NASA aerospace safety and engineering queries: aviation safety, aircraft icing, flight control fault tolerance, cockpit human factors, runway safety, turbine engines, aerodynamic stall, avionics software verification, thermal protection, structural fatigue, air traffic management, unmanned aircraft, rocket propulsion, Space Shuttle safety, wind shear/turbulence, composite structures, spaceflight life support, hypersonic aerodynamics, rotorcraft, and reliability/hazard analysis.
"""
    (out_dir / "README.md").write_text(card, encoding="utf-8")

    summary = {
        "repo_id": repo_id,
        "documents": n_docs,
        "parents": n_parents,
        "children": n_children,
        "eval": n_eval,
        "total_rows": total_rows,
        "total_mb": round(total_mb, 2),
        "kept_copyright": dict(kept_counts),
        "excluded_copyright": dict(excluded_counts),
    }
    (out_dir / "release_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Release built cleanly: %s", summary)
    return summary


def upload_release(out_dir: Path, repo_id: str) -> str:
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True, private=False)
    logger.info("Uploading %s to https://huggingface.co/datasets/%s ...", out_dir, repo_id)
    api.upload_folder(
        folder_path=str(out_dir),
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Upload license-filtered NASA NTRS RAG Parquet corpus (documents, parents, children, eval)",
    )
    url = f"https://huggingface.co/datasets/{repo_id}"
    logger.info("Published dataset to %s", url)
    return url


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="hsilvosa/nasa-ntrs-aerospace")
    parser.add_argument("--out-dir", type=Path, default=Path("D:/datasets/nasa-ntrs/hf_release"))
    parser.add_argument("--no-upload", action="store_true")
    args = parser.parse_args()

    build_release(args.out_dir, args.repo_id)
    if not args.no_upload:
        upload_release(args.out_dir, args.repo_id)


if __name__ == "__main__":
    main()

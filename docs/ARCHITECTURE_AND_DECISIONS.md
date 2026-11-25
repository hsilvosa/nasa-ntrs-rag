# Architecture, Components & Engineering Decisions

This document explains how **NASA NTRS Aerospace Safety RAG** was built, what each module in the repository does, why each architectural choice was made, and what empirical evidence changed the design when scaling from 31 reports (346 MB) to **2,426 reports (10.01 GB, 241,641 child chunks)** on a single NVIDIA RTX 3060 (12 GB VRAM).

---

## 1. Development Chronology

1. **v0 Baseline & Core Scaffold:**
   - Built a framework-free modular Python package (`src/nasa_ntrs_rag`) with Pydantic schemas, fixed-size baseline chunking (`fixed_chunk_document`), hierarchical parent-child chunking (`hierarchical_chunk_document`), PyTorch + Transformers BGE embeddings and cross-encoder reranking, Qdrant + BM25 hybrid retrieval, deterministic query routing, abstention gating, FastAPI + Inspector UI, Typer CLI, and unit tests (`tests/test_core.py`).
2. **31-Report Validation Run (345.77 MB → 4,894 parents / 14,940 children):**
   - Installed CUDA PyTorch (`torch 2.11.0+cu128`) and ran the full offline pipeline (`download → ingest → index → eval-ablation`).
   - First ablation exposed a subtle contextual-retrieval bug: `hybrid_rerank` dropped 20.2 MRR@10 points (`0.717` vs `0.919`) because the cross-encoder received bare `chunk.text` instead of `contextual_text`. Fixing that immediately restored `MRR@10 = 0.919` while holding **100.0% out-of-domain abstention**.
3. **10 GB Scale-Up (2,426 Reports / 10.01 GB → 77,379 parents / 241,641 children):**
   - Upgraded the NTRS client with paginated search (`page[size]=100`), 20 balanced aerospace domains in round-robin with persistent candidate queues, per-file 80 MB caps, `HTTP 429/503` retries, and a total `--max-gb 10.0` budget cap.
   - Hit a severe PyMuPDF parsing wall (>5 hours projected with unconditional `page.find_tables()` on vector-heavy aerodynamic plots). Adding selective table detection (`Table N` regex + `4 <= drawings <= 400`) sped parsing up **~34–78×**, completing all 2,426 PDFs in ~9 minutes across 6 processes.
   - Optimized GPU indexing with length-sorted `fp16` batching (`batch_size=64`, sustaining 99% RTX 3060 utilization at ~164 chunks/s), 8,192-point streaming upserts to avoid embedded Qdrant `O(N^2)` reallocations, disk-cached BM25 (`bm25.pkl`), and `numpy.argpartition` sparse top-K selection.
   - Ran a 105-query stratified ablation across all 20 domains (`eval_100.jsonl`), proving that Hybrid (Dense + BM25 RRF) decisively outperforms dense-only search at 241k chunks (`+3.0` Recall@5, `+5.8` MRR@10, `+4.8` nDCG@10).
4. **Static Web Demo Exporter (`scripts/export_web_demo.py`):**
   - Exported 12 real end-to-end GPU + local `Qwen3-4B` traces (`traces.json`) alongside a compact 2,426-report client-side BM25 search index (`corpus.json`) so the system can be inspected live on a static site with strict Content Security Policy and zero server cost.

---

## 2. System Overview & Module Map ("What Everything Is")

```text
NASA NTRS Open API (2,426 PDFs, 10.01 GB)
        │
        ▼
[ingestion/ntrs_client.py]  Paginated, concurrent round-robin downloader -> manifest.jsonl
        │
        ▼
[ingestion/parser.py]       Layout-aware PDF -> Markdown (PyMuPDF selective tables / Docling)
        │
        ▼
[ingestion/chunker.py]      Hierarchical Parent (~4,800 ch) & Child (~1,200 ch) + Contextual Headers
        │
        ▼
[ingestion/pipeline.py]     Multiprocess PDF parser + JSONL serializer (parents.jsonl / children.jsonl)
        │
        ├─────────────────────────────────────────────┐
        ▼                                             ▼
[retrieval/embeddings.py]                    [retrieval/hybrid_store.py]
DenseEmbedder (BGE fp16 CUDA)                BM25Okapi (technical tokenizer + bm25.pkl)
        │                                             │
        ▼                                             │
Qdrant Vector Store (241,641 vectors)                 │
        │                                             │
        └───────── Reciprocal Rank Fusion (k=60) ─────┘
                               │
                               ▼
                 [retrieval/query_router.py]
                 Deterministic filters (Year, NASA Center, Doc ID)
                               │
                               ▼
                 [retrieval/embeddings.py]
                 CrossEncoderReranker (bge-reranker-v2-m3, sigmoid [0,1])
                               │
                               ▼
                 [generation/rag_pipeline.py]
                 Abstention Gate (< 0.15 => 0 ms pre-LLM block)
                 Parent Expansion & Deduplication -> Strict Citation Prompt
                               │
                               ▼
                 [generation/llm.py]
                 Pluggable LLM: Local llama-server GGUF (Qwen3-4B) or Gemini API
```

### Detailed Module Reference

| File | Responsibility | Key Implementation Details |
| :--- | :--- | :--- |
| `src/nasa_ntrs_rag/config.py` | Centralized configuration | Pydantic `BaseSettings` reading `.env`. Keeps all heavy datasets (`raw/`, `processed/`, `qdrant/`, `eval/`) outside git in `DATA_DIR` (`D:\datasets\nasa-ntrs`). Resolves `DEVICE=auto` to `cuda` when available. |
| `src/nasa_ntrs_rag/schemas.py` | Typed data contracts | Pydantic v2 models: `NTRSDocumentMetadata`, `ParentChunk`, `ChildChunk`, `QueryFilters`, `RetrievedChunk`, `Citation`, `StageTimingsMs`, `RAGResponse`, `EvalSample`. |
| `src/nasa_ntrs_rag/ingestion/ntrs_client.py` | Corpus acquisition | Talks to `https://ntrs.nasa.gov/api/citations`. Uses round-robin pagination (`page[size]=100`) across 20 aerospace domains, persistent per-topic queues, `%PDF` magic-byte verification, `max_pdf_mb` caps, `HTTP 429/503` retries, and `max_total_gb` stops. |
| `src/nasa_ntrs_rag/ingestion/parser.py` | Layout-aware PDF parsing | Converts PDFs to Markdown with `<!-- page: N -->` markers. Promotes numbered/uppercase NASA section headings (`## 2.1 Results`) and extracts GFM tables selectively (`Table N` hint + `4 <= drawings <= 400`). Supports `pymupdf`, `docling`, `plain`, and `auto`. |
| `src/nasa_ntrs_rag/ingestion/chunker.py` | Hierarchical chunking | Splits on Markdown headings while tracking page ranges. `_split_preserving_tables` never slices inside a Markdown table. Injects `[NASA NTRS <id> \| <title> \| Year \| Center \| Section]` into `contextual_text`. Also preserves `fixed_chunk_document` for v0 ablation comparisons. |
| `src/nasa_ntrs_rag/ingestion/pipeline.py` | Offline ingestion orchestrator | Coordinates `ProcessPoolExecutor` PDF parsing (reusing cached `.md` files), logs per-document parse failures without aborting, and writes `parents.jsonl` and `children.jsonl`. |
| `src/nasa_ntrs_rag/retrieval/embeddings.py` | Neural bi-encoder & cross-encoder | Direct PyTorch + Hugging Face Transformers (`torch.inference_mode()`, `dtype=torch.float16` on CUDA). `DenseEmbedder` uses CLS pooling for BGE, query-prefix formatting, L2 normalization, and length-sorted batching. `CrossEncoderReranker` computes sigmoid relevance probabilities in `[0, 1]`. |
| `src/nasa_ntrs_rag/retrieval/query_router.py` | Deterministic metadata filter extraction | Zero-LLM regex router extracting year ranges (`between 2010 and 2024`, `since 2018`, `before 2005`), canonical NASA center names (`Glenn`, `Langley`, `Ames`, `Armstrong`/`Dryden`, `Johnson`, `Marshall`, `Kennedy`, `Goddard`, `JPL`), and 11-digit NTRS IDs. |
| `src/nasa_ntrs_rag/retrieval/hybrid_store.py` | Hybrid retrieval engine | Manages Qdrant (embedded local or remote Docker URL), BM25Okapi with technical tokenization (`[A-Za-z0-9\-_.]+`), `bm25.pkl` cache invalidation, `np.argpartition` top-K sparse search, Reciprocal Rank Fusion (`rrf_k=60`), contextual cross-encoder reranking, and parent-section deduplication. |
| `src/nasa_ntrs_rag/generation/llm.py` | Pluggable LLM layer | `BaseLLMClient` protocol implemented by `OpenAICompatibleLLM` (local `llama-server`, Ollama, OpenAI, Groq) and `GeminiLLM` (`google-genai`). Supports both `generate()` and `stream()`. |
| `src/nasa_ntrs_rag/generation/rag_pipeline.py` | Grounded RAG orchestrator | Runs query routing → retrieval → pre-LLM abstention gate (`top_score < 0.15`) → numbered parent context formatting → strict citation prompt → `RAGResponse` with per-stage millisecond telemetry. |
| `src/nasa_ntrs_rag/evaluation/metrics.py` | IR benchmark suite | Computes `Recall@K`, `MRR@K`, `nDCG@K`, unanswerable abstention rate, and p50/p95 latencies across `dense`, `hybrid`, and `hybrid_rerank`. Stratifies factual queries round-robin across topics when sampling from large manifests. |
| `src/nasa_ntrs_rag/api/main.py` & `inspector.html` | Local HTTP API & UI | FastAPI server exposing `/health`, `/stats`, `/query`, and a zero-dependency live Inspector UI at `/` showing per-stage scores, citations, and latency breakdown. |
| `src/nasa_ntrs_rag/cli.py` | Unified CLI (`ntrs-rag`) | Typer commands: `download`, `ingest`, `index`, `ask`, `eval-ablation`, `serve`. |
| `scripts/serve_local_llm.ps1` | Local GGUF launcher | Starts `llama-server` on `127.0.0.1:8080/v1` with `--flash-attn on` and full GPU offload (`--n-gpu-layers 99`). |
| `scripts/export_web_demo.py` | Static portfolio exporter | Runs 12 curated queries × 3 retrieval modes against the real GPU index + local LLM and packages `traces.json` + the 2,426-document `corpus.json` for zero-backend browser inspection. |

---

## 3. Architectural Decisions (ADRs)

### ADR-01: Zero Black-Box Frameworks (Direct PyTorch + Transformers)
- **Context:** Frameworks like LangChain or LlamaIndex abstract away tokenization, pooling, batch sorting, metadata filter parity, and reranker inputs.
- **Decision:** Implement `DenseEmbedder`, `CrossEncoderReranker`, RRF, BM25, and the RAG pipeline directly in ~1,200 lines of typed Python.
- **Trade-off:** Slightly more initial code, in exchange for complete transparency. This directly enabled catching the contextual-header reranker bug (+20.2 MRR) and implementing length-sorted fp16 batching + 8,192-point streaming upserts in minutes.

### ADR-02: Hierarchical Parent-Child Chunking with Contextual Headers
- **Context:** Fixed 1,000-character chunks frequently split tables and strip document/section context, causing retrieval confusion across multi-volume reports (e.g. *Columbia* Volumes I–VI).
- **Decision:** Retrieve fine-grained child chunks (~1,200 chars) enriched with `[NASA NTRS <id> | <title> | Year | Center | Section]`, then expand to the enclosing parent section (~4,800 chars) and deduplicate by `parent_id` before prompting the LLM.
- **Evidence:** Passing `contextual_text` consistently to Qdrant, BM25, **and** the cross-encoder reranker raised 31-doc `hybrid_rerank` MRR@10 from `0.717` to `0.919` (+20.2 pts).

### ADR-03: Round-Robin Paginated Download with GB Budget Cap
- **Context:** Sequential per-topic downloading with a byte budget lets the first few topics (which contain 150 MB scanned volumes) exhaust the disk quota before reaching later domains.
- **Decision:** Maintain persistent per-topic candidate queues, paginate `page[size]=100`, download in controlled round-robin waves across all 20 domains, skip individual PDFs >80 MB, and retry `HTTP 429/503` once.
- **Evidence:** Collected **2,426 valid PDFs (10.01 GB)** in ~35 minutes with balanced representation (85–116 docs across primary topics) and resumable checkpointing.

### ADR-04: Selective Table Extraction in PyMuPDF
- **Context:** Calling `page.find_tables()` on every page of 2,395 NASA PDFs stalled at 75 PDFs / 10 minutes (>5 hours projected) due to CAD drawings and dense scatter plots containing thousands of vector paths.
- **Decision:** Run `page.find_tables()` only on pages that match `\btable\s+[0-9IVX]+` and have `4 <= len(page.get_drawings()) <= 400`.
- **Evidence:** Speed increased from ~0.12 PDFs/s (12 workers) to **9.7 PDFs/s single-threaded** (34–78× faster), parsing all 2,426 PDFs in ~9 minutes across 6 workers while retaining 65.5 KB average Markdown per document.

### ADR-05: Hybrid Dense + Technical BM25 via Reciprocal Rank Fusion
- **Context:** Aerospace reports rely heavily on exact identifiers (`NASA-TM-20220007001`, `B737-MAX`, `AAMP-FV`, `RUNSAFE`), which dense embeddings compress and standard English tokenizers split apart.
- **Decision:** Pair `BAAI/bge-base-en-v1.5` with a custom regex BM25 tokenizer (`[A-Za-z0-9\-_.]+`) fused via RRF (`k = 60`), applying identical `QueryFilters` to both retrievers.
- **Evidence:** On 31 reports, dense search alone scored `0.968` Recall@5 and masked BM25's value. At **241,641 chunks (2,426 reports)**, `hybrid` beat `dense` on every ranking metric: `Recall@5` `0.920 → 0.950`, `MRR@10` `0.849 → 0.907`, `nDCG@10` `0.870 → 0.918`.

### ADR-06: Length-Sorted GPU Batching, Streaming Upserts & Disk-Cached BM25
- **Context:** Naive embedding + upsert of 241,641 chunks causes heavy padding overhead on GPU, high Python peak RAM, repeated `O(N^2)` matrix reallocations in local embedded Qdrant, and ~40 s BM25 rebuilds on every CLI invocation.
- **Decision:**
  - Sort texts by length inside `DenseEmbedder.encode()` and restore original order after `fp16` inference (`batch_size=64`).
  - Stream embeddings and Qdrant upserts in 8,192-point blocks for local mode (256 for remote HTTP) and store only filterable metadata in Qdrant payload (avoiding duplicating 300+ MB of chunk text already stored in `children.jsonl`).
  - Persist `BM25Okapi` to `bm25.pkl` keyed by `(file_size, mtime_ns)` and use `numpy.argpartition` for unfiltered sparse top-K.
- **Evidence:** Sustained **99% RTX 3060 utilization (~164 chunks/s, 5.7 GB VRAM)** indexing all 241,641 chunks in ~27 minutes without OOM.

### ADR-07: Pre-LLM Calibrated Abstention Gate
- **Context:** Relying purely on LLM prompt instructions to refuse out-of-domain questions wastes 1.5–3.5 s of generation latency and risks hallucination when irrelevant passages look plausible.
- **Decision:** Gate generation on the top sigmoid probability of `BAAI/bge-reranker-v2-m3` (`ABSTENTION_SCORE_THRESHOLD=0.15`).
- **Evidence:** Achieved **100.0% out-of-domain abstention** at both 14.9k and 241.6k chunks (`0.0005`, `0.0172`, and `0.0546` on recorded OOD queries → `0 ms` generation cost), and rescued the metadata-filtered Glenn icing (`2010–2024`) query where `dense` and `hybrid` failed to surface an answerable passage.
- **Known Trade-off:** At 2,426 reports, pure reranker replacement lowered strict doc-ID MRR@10 from `0.907` (`hybrid`) to `0.749` because the cross-encoder occasionally preferred explanatory passages from sibling reports of the same NASA study. Combining RRF rank with reranker score is the recommended next step.

### ADR-08: Embedded Qdrant First, Docker Qdrant for Production Latency
- **Context:** Embedded local Qdrant (`QdrantClient(path=...)`) requires zero external services and simplifies reproducibility on a workstation, but executes brute-force cosine search in memory and warns past 20,000 points.
- **Decision:** Keep embedded local Qdrant as the zero-setup default while shipping `docker-compose.yml` (`QDRANT_URL=http://localhost:6333`) for production HNSW indexing.
- **Evidence:** Measured dense p50 rose from `59.7 ms` (14.9k chunks) to `829.8 ms` (241.6k chunks), and pure-Python BM25 added ~2.3 s—quantifying the exact threshold where moving to Qdrant server (HNSW + native Rust sparse vectors) pays off.

### ADR-09: Static Browser Demo Architecture
- **Context:** Hosting 10 GB of vectors plus two transformer models and a 4B LLM continuously for a portfolio site incurs ongoing GPU server costs or 30–60 s cold starts, and violates strict same-origin CSP (`connect-src 'self'`).
- **Decision:** Precompute 12 real RTX 3060 + `Qwen3-4B` traces across all 3 retrieval modes (`traces.json`, 514 KB) and pair them with a live browser-side technical BM25 + query-router search engine over all 2,426 report records (`corpus.json`, 1.33 MB).
- **Evidence:** Loads instantaneously on a static Cloudflare Workers site, passes strict CSP and privacy checks, and shows real GPU scores, parent expansions, citations, and abstentions honestly labeled as recorded runs.

---

## 4. Storage Layout (`DATA_DIR=D:\datasets\nasa-ntrs`)

```text
D:\datasets\nasa-ntrs\
├── raw\
│   ├── pdfs\                  # 2,426 NASA NTRS PDFs (10.01 GB)
│   └── metadata\
│       ├── manifest.jsonl     # Normalized metadata for all 2,426 docs
│       └── <doc_id>.json      # Raw NTRS API JSON responses
├── processed\
│   ├── markdown\              # 2,426 cached Markdown files with <!-- page: N -->
│   └── chunks\
│       ├── parents.jsonl      # 77,379 section-level parent chunks
│       ├── children.jsonl     # 241,641 contextual child chunks
│       └── bm25.pkl           # Disk-cached BM25Okapi index
├── qdrant\                    # Embedded local Qdrant collection (nasa_ntrs_chunks)
└── eval\
    ├── starter_eval.jsonl     # 36-query eval set (31-doc corpus)
    ├── eval_100.jsonl         # 105-query stratified eval set (2,426-doc corpus)
    └── ablation_results.json  # Latest measured IR & latency report
```

---

## 5. Production Hardening Roadmap

1. **Docker Qdrant HNSW + Native Sparse Vectors:** Switch `QDRANT_URL=http://localhost:6333` and move BM25/SPLADE into Qdrant's Rust sparse engine to bring 241k-chunk hybrid latency from ~3.1 s down to <150 ms.
2. **RRF + Reranker Score Fusion:** Blend normalized RRF rank with cross-encoder probability so `hybrid_rerank` retains **100% OOD abstention** without conceding doc-ID MRR@10 on sibling NASA reports.
3. **Curated Human Golden Set + Faithfulness Judge:** Add 25–30 hand-written multi-document engineering questions requiring exact numerical/table values, paired with automated claim-to-citation faithfulness scoring.
4. **OCR Fallback for Scanned Legacy Reports:** Run the optional `docling` extra specifically on the 6 legacy image-only PDFs that returned empty text under PyMuPDF.

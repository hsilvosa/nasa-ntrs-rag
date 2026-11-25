# NASA NTRS Aerospace Safety RAG

Production-grade **Hybrid & Hierarchical Retrieval-Augmented Generation (RAG)** system over open-access **NASA Technical Reports Server (NTRS)** aerospace safety and engineering reports, built for reproducible IR benchmarking and strict source grounding.

## Architecture

```text
NASA NTRS Open API (PDF + Metadata JSON -> D:\datasets\nasa-ntrs\raw)
        │
        ▼
Layout-Aware PDF Parser (Docling / PyMuPDF + Markdown Tables + Page Markers)
        │
        ▼
Hierarchical Parent-Child Chunker + Contextual Headers
        │
        ▼
Qdrant (Dense BGE Embeddings + Metadata Filters)  +  BM25 Technical Tokenizer
        │                                                    │
        └────────────────── Reciprocal Rank Fusion (RRF) ────┘
                                      │
                                      ▼
                   Cross-Encoder Reranker (bge-reranker-v2-m3)
                                      │
                                      ▼
               Parent Expansion + Confidence Abstention Gate
                                      │
                                      ▼
       Pluggable LLM (Local llama-server GGUF / Gemini / OpenAI) -> Cited Answer
```

## Key Engineering Highlights

- **Zero Black-Box Abstractions:** Clean PyTorch + Hugging Face Transformers bi-encoder (`DenseEmbedder`) and cross-encoder (`CrossEncoderReranker`) with CUDA `fp16` acceleration (tuned for RTX 3060 12 GB).
- **Layout-Aware Parsing & Parent-Child Hierarchy:** Preserves section boundaries, exact page numbers, and Markdown tables intact; retrieves fine-grained child chunks and expands to parent sections for causal context.
- **Hybrid Search + Deterministic Query Routing:** Combines Qdrant cosine search with BM25 lexical matching (preserving aerospace codes like `NASA-TM-20210012345` or `B737-MAX`) via Reciprocal Rank Fusion (RRF) and automatic temporal/center/doc-ID filter extraction.
- **Calibrated Abstention & Inline Citations:** Blocks out-of-corpus queries before LLM generation when the top cross-encoder relevance score falls below threshold.
- **Pluggable Local & Cloud LLMs:** Switch seamlessly between local GGUF models via `llama-server` (`D:\models`) and cloud APIs (Gemini / OpenAI-compatible).
- **Reproducible IR Ablation Suite:** Measures `Recall@5`, `MRR@10`, `nDCG@10`, unanswerable abstention rate, and p50/p95 latency across `dense`, `hybrid`, and `hybrid_rerank`.

## Setup

Copy `.env.example` to `.env` (already configured for `DATA_DIR=D:/datasets/nasa-ntrs`):

```powershell
Copy-Item .env.example .env
```

Install dependencies into an isolated project virtual environment with `uv`:

```powershell
uv sync --extra ml --extra dev
# Optional: Docling layout parser & Gemini SDK
uv sync --extra ml --extra docling --extra gemini --extra dev
```

*(Alternatively, run inside your existing CUDA conda environment `wuxia` after installing the lightweight RAG packages: `qdrant-client`, `rank-bm25`, `pymupdf`, `typer`, `rich`.)*

## End-to-End Quickstart

```powershell
# 1. Download NASA NTRS aerospace safety PDFs + metadata to D:\datasets\nasa-ntrs
ntrs-rag download --per-topic 4 --max-docs 24

# 2. Parse PDFs to structured Markdown & build Parent-Child chunks
ntrs-rag ingest --backend auto --strategy hierarchical

# 3. Embed chunks on GPU and build the Qdrant + BM25 hybrid index
ntrs-rag index --strategy hierarchical

# 4. Run reproducible retrieval ablation (Dense vs Hybrid vs Hybrid+Rerank)
ntrs-rag eval-ablation

# 5. (Optional) Start a local GGUF model from D:\models via llama-server
.\scripts\serve_local_llm.ps1 -Model qwen3-4b

# 6. Query via CLI or launch the FastAPI + Live Inspector UI (http://127.0.0.1:8001)
ntrs-rag ask "What aerodynamic effects does supercooled droplet icing have on wing stall?"
ntrs-rag serve --port 8001
```

## Retrieval Ablation Results

Measured on **31 NASA NTRS aerospace safety reports** (345.8 MB PDFs → 4,894 parent chunks, 14,940 child chunks) across **36 evaluation queries** (31 factual + 5 out-of-domain unanswerable controls) on an NVIDIA RTX 3060 12 GB (`BAAI/bge-base-en-v1.5` + `BAAI/bge-reranker-v2-m3`, CUDA `fp16`, `ABSTENTION_SCORE_THRESHOLD=0.15`):

| Mode | Recall@5 | MRR@10 | nDCG@10 | Unanswerable Abstention | Latency p50 (ms) | Latency p95 (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `dense` (Baseline) | 0.968 | 0.919 | 0.932 | 0.0% | 59.7 | 66.9 |
| `hybrid` (Dense + BM25 RRF) | 0.968 | 0.887 | 0.907 | 0.0% | 154.9 | 198.5 |
| `hybrid_rerank` (+ bge-reranker-v2-m3) | **0.968** | **0.919** | **0.932** | **100.0%** | 561.7 | 694.1 |

## Testing & Linting

```powershell
pytest
ruff check src tests
```

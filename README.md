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

*(For a full component-by-component breakdown, development timeline, and Architectural Decision Records with measured evidence, see [docs/ARCHITECTURE_AND_DECISIONS.md](docs/ARCHITECTURE_AND_DECISIONS.md). A license-filtered Parquet release of the parsed Markdown, parent/child chunks, and evaluation set — 2,317 verified public-use reports, 308,624 rows, 226.5 MB — is published at [huggingface.co/datasets/hsilvosa/nasa-ntrs-aerospace](https://huggingface.co/datasets/hsilvosa/nasa-ntrs-aerospace).)*

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

Measured on **2,426 NASA NTRS aerospace safety & engineering reports** (10.01 GB PDFs across 20 NASA domains → **77,379 parent chunks**, **241,641 child chunks**) across **105 stratified evaluation queries** (100 factual + 5 out-of-domain unanswerable controls) on an NVIDIA RTX 3060 12 GB (`BAAI/bge-base-en-v1.5` + `BAAI/bge-reranker-v2-m3`, CUDA `fp16`, embedded Qdrant + BM25, `ABSTENTION_SCORE_THRESHOLD=0.15`):

| Mode | Recall@5 | MRR@10 | nDCG@10 | Unanswerable Abstention | Latency p50 (ms) | Latency p95 (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `dense` (Baseline) | 0.920 | 0.849 | 0.870 | 0.0% | 829.8 | 1007.9 |
| `hybrid` (Dense + BM25 RRF) | **0.950** | **0.907** | **0.918** | 0.0% | 3130.6 | 4301.7 |
| `hybrid_rerank` (+ bge-reranker-v2-m3) | 0.930 | 0.749 | 0.800 | **100.0%** | 3609.2 | 4351.5 |

> **Scale takeaways:** Moving from 31 reports (14.9k chunks) to 2,426 reports (241.6k chunks) makes the value of hybrid lexical+dense fusion clear (`+3.0` pts Recall@5, `+5.8` pts MRR@10, `+4.8` pts nDCG@10 over dense-only), while the cross-encoder gate achieves **100% abstention** on out-of-domain queries. For sub-50 ms dense search at >20k chunks, run Qdrant via `docker compose up -d` and set `QDRANT_URL=http://localhost:6333`.

## Testing & Linting

```powershell
pytest
ruff check src tests
```

"""Centralized application settings loaded from environment / .env."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Storage paths (defaults to D:\datasets\nasa-ntrs)
    data_dir: Path = Field(default=Path("D:/datasets/nasa-ntrs"))

    # Vector DB
    qdrant_url: str | None = Field(default=None)
    qdrant_collection: str = Field(default="nasa_ntrs_chunks")

    # Embedding & Reranker models
    embedding_model: str = Field(default="BAAI/bge-base-en-v1.5")
    reranker_model: str = Field(default="BAAI/bge-reranker-v2-m3")
    device: Literal["auto", "cuda", "cpu"] = Field(default="auto")
    embedding_batch_size: int = Field(default=32)

    # Chunking parameters (Parent-Child hierarchy)
    child_chunk_chars: int = Field(default=1200)
    child_chunk_overlap: int = Field(default=200)
    parent_chunk_chars: int = Field(default=4800)

    # Retrieval & Grounding parameters
    retrieval_top_k_dense: int = Field(default=25)
    retrieval_top_k_sparse: int = Field(default=25)
    retrieval_top_k_rerank: int = Field(default=5)
    rrf_k: int = Field(default=60)
    abstention_score_threshold: float = Field(default=0.15)

    # LLM configuration
    llm_provider: Literal["openai_compatible", "gemini"] = Field(default="openai_compatible")
    llm_base_url: str = Field(default="http://127.0.0.1:8080/v1")
    llm_api_key: str = Field(default="local-no-key-needed")
    gemini_api_key: str | None = Field(default=None)
    llm_model: str = Field(default="qwen3-4b-instruct")
    llm_temperature: float = Field(default=0.1)
    llm_max_tokens: int = Field(default=1024)

    # NASA NTRS API
    ntrs_api_base: str = Field(default="https://ntrs.nasa.gov/api/citations")
    ntrs_request_timeout: float = Field(default=60.0)

    @property
    def raw_pdf_dir(self) -> Path:
        return self.data_dir / "raw" / "pdfs"

    @property
    def raw_metadata_dir(self) -> Path:
        return self.data_dir / "raw" / "metadata"

    @property
    def manifest_path(self) -> Path:
        return self.raw_metadata_dir / "manifest.jsonl"

    @property
    def processed_md_dir(self) -> Path:
        return self.data_dir / "processed" / "markdown"

    @property
    def processed_chunks_dir(self) -> Path:
        return self.data_dir / "processed" / "chunks"

    @property
    def qdrant_local_path(self) -> Path:
        return self.data_dir / "qdrant"

    @property
    def eval_dir(self) -> Path:
        return self.data_dir / "eval"

    def ensure_directories(self) -> None:
        """Create all required data directories under data_dir."""
        for path in (
            self.raw_pdf_dir,
            self.raw_metadata_dir,
            self.processed_md_dir,
            self.processed_chunks_dir,
            self.qdrant_local_path,
            self.eval_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def resolve_device(self) -> str:
        """Resolve 'auto' to 'cuda' if available, else 'cpu'."""
        if self.device != "auto":
            return self.device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

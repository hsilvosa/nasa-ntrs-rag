"""PyTorch + Transformers dense embedder and cross-encoder reranker (CUDA-accelerated)."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)

_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class DenseEmbedder:
    """Bi-encoder embedder using Hugging Face Transformers with CUDA fp16 support."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-base-en-v1.5",
        device: str = "cuda",
        batch_size: int = 32,
        max_length: int = 512,
    ) -> None:
        self.model_name = model_name
        self.device = device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
        self.batch_size = batch_size
        self.max_length = max_length
        self._use_cls_pooling = "bge" in model_name.lower()
        self._is_bge_en = "bge-" in model_name.lower() and "m3" not in model_name.lower()

        logger.info("Loading embedding model %s on %s", model_name, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModel.from_pretrained(model_name, dtype=dtype).to(self.device)
        self.model.eval()
        self.dimension: int = int(self.model.config.hidden_size)

    def _pool(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self._use_cls_pooling:
            return last_hidden_state[:, 0]
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        summed = torch.sum(last_hidden_state * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts

    @torch.inference_mode()
    def encode(self, texts: Sequence[str], is_query: bool = False) -> np.ndarray:
        """Encode a sequence of texts into L2-normalized float32 embeddings."""
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)

        prepared = [
            f"{_BGE_QUERY_PREFIX}{t}" if (is_query and self._is_bge_en) else t for t in texts
        ]
        n = len(prepared)
        # Sort by character length for large batches to minimize transformer padding waste
        if n > self.batch_size:
            order = sorted(range(n), key=lambda idx: len(prepared[idx]))
            ordered_texts = [prepared[i] for i in order]
        else:
            order = None
            ordered_texts = prepared

        all_vecs: list[np.ndarray] = []
        for start in range(0, n, self.batch_size):
            batch = ordered_texts[start : start + self.batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            outputs = self.model(**encoded)
            pooled = self._pool(outputs.last_hidden_state, encoded["attention_mask"])
            normalized = F.normalize(pooled.float(), p=2, dim=1)
            all_vecs.append(normalized.cpu().numpy().astype(np.float32))

        stacked = np.vstack(all_vecs)
        if order is not None:
            restored = np.empty_like(stacked)
            restored[order] = stacked
            return restored
        return stacked


class CrossEncoderReranker:
    """Cross-encoder reranker (e.g. BAAI/bge-reranker-v2-m3) with sigmoid probability scores."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: str = "cuda",
        batch_size: int = 16,
        max_length: int = 512,
    ) -> None:
        self.model_name = model_name
        self.device = device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
        self.batch_size = batch_size
        self.max_length = max_length

        logger.info("Loading reranker model %s on %s", model_name, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, dtype=dtype
        ).to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        """Compute relevance scores in [0, 1] for (query, passage) pairs."""
        if not passages:
            return []

        scores: list[float] = []
        for start in range(0, len(passages), self.batch_size):
            batch_passages = list(passages[start : start + self.batch_size])
            pairs = [[query, p] for p in batch_passages]
            inputs = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            logits = self.model(**inputs, return_dict=True).logits.view(-1).float()
            probs = torch.sigmoid(logits)
            scores.extend(probs.cpu().tolist())
        return scores

"""Configurable local dense embeddings. No model is downloaded on import."""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np

from app.rag.schemas import KnowledgeChunk


class EmbeddingError(RuntimeError):
    """Model loading or encoding failed; never substitute fabricated vectors."""


@dataclass(frozen=True)
class EmbeddingConfig:
    model_name_or_path: str
    revision: str | None = None
    device: str = "cpu"
    batch_size: int = 16
    local_files_only: bool = True
    normalize: bool = True
    query_prefix: str = ""
    document_prefix: str = ""

    def __post_init__(self):
        if not self.model_name_or_path.strip():
            raise ValueError("An embedding model name or local path is required.")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive.")

    @classmethod
    def from_env(cls) -> EmbeddingConfig:
        # Environment variables must be exported by the caller; no .env secrets
        # are read or logged by this module.
        return cls(
            model_name_or_path=os.environ.get("EMBEDDING_MODEL", ""),
            revision=os.environ.get("EMBEDDING_REVISION") or None,
            device=os.environ.get("EMBEDDING_DEVICE", "cpu"),
            batch_size=int(os.environ.get("EMBEDDING_BATCH_SIZE", "16")),
            query_prefix=os.environ.get("EMBEDDING_QUERY_PREFIX", ""),
            document_prefix=os.environ.get("EMBEDDING_DOCUMENT_PREFIX", ""),
        )


def chunk_embedding_text(chunk: KnowledgeChunk) -> str:
    """Add retrieval context without changing stored chunk text or its ID."""
    parts = [chunk.metadata.title]
    if chunk.metadata.section and chunk.metadata.section != chunk.metadata.title:
        parts.append(chunk.metadata.section)
    parts.append(chunk.text)
    return "\n\n".join(parts)


class EmbeddingProvider(ABC):
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Vector size for both document and query embeddings."""

    @property
    @abstractmethod
    def info(self) -> dict:
        """Model identity/configuration to retain alongside indexed vectors."""

    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        """Encode a query using the same model as document embeddings."""

    @abstractmethod
    def embed_chunks(self, chunks: Sequence[KnowledgeChunk]) -> list[list[float]]:
        """Return one vector per input chunk, in the same order."""


def _load_model(config: EmbeddingConfig):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(
        config.model_name_or_path,
        revision=config.revision,
        device=config.device,
        local_files_only=config.local_files_only,
        trust_remote_code=False,
    )


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    """One replaceable Sentence Transformers encoder, loaded once per instance."""

    def __init__(self, config: EmbeddingConfig):
        self.config = config
        try:
            self._model = _load_model(config)
            self._dimension = int(self._model.get_embedding_dimension())
            if self._dimension < 1:
                raise ValueError("Invalid model dimension.")
        except Exception as exc:
            raise EmbeddingError("Unable to load configured local embedding model.") from exc

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def info(self) -> dict:
        return {"provider": "sentence_transformers", **asdict(self.config),
                "dimension": self.dimension}

    def _encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if any(not text.strip() for text in texts):
            raise ValueError("Embedding text must not be empty.")
        try:
            # Reject truncation: the source excerpt must correspond to all text
            # embedded. Chunk sizing itself remains the responsibility of Phase 1.
            tokenized = self._model.tokenizer(
                texts, truncation=False, padding=False, add_special_tokens=True,
            )
            if any(len(ids) > self._model.max_seq_length for ids in tokenized["input_ids"]):
                raise EmbeddingError("Input exceeds the configured model token limit.")
            vectors = np.asarray(self._model.encode(
                texts, batch_size=self.config.batch_size,
                convert_to_numpy=True, normalize_embeddings=self.config.normalize,
                show_progress_bar=False, prompt="",
            ), dtype=np.float32)
            if vectors.shape != (len(texts), self.dimension):
                raise EmbeddingError("Embedding output shape does not match inputs.")
            if not np.isfinite(vectors).all():
                raise EmbeddingError("Embedding output contains non-finite values.")
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            if np.any(norms == 0):
                raise EmbeddingError("Embedding output contains zero vectors.")
            if self.config.normalize:
                vectors = vectors / norms
            return vectors.tolist()
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError("Embedding encoding failed.") from exc

    def embed_text(self, text: str) -> list[float]:
        if not text.strip():
            raise ValueError("Embedding text must not be empty.")
        return self._encode([self.config.query_prefix + text])[0]

    def embed_chunks(self, chunks: Sequence[KnowledgeChunk]) -> list[list[float]]:
        return self._encode([
            self.config.document_prefix + chunk_embedding_text(chunk)
            for chunk in chunks
        ])

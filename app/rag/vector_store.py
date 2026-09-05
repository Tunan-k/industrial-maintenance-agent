"""Backend-independent vector storage contract; no query text or retrieval policy."""
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Sequence

from app.rag.schemas import KnowledgeChunk


class VectorStoreError(RuntimeError):
    """Vector database operation or payload validation failed."""


def validate_metadata_filters(filters: Mapping[str, str] | None) -> dict[str, str]:
    """Phase 3 supports exact equipment-type matching; never ignore bad keys."""
    if filters is None:
        return {}
    if not isinstance(filters, Mapping):
        raise ValueError("filters must be a metadata mapping.")
    for key, value in filters.items():
        if key != "equipment_type":
            raise ValueError(f"Unsupported metadata filter: {key}")
        if not isinstance(value, str) or not value.strip():
            raise ValueError("equipment_type filter must be a non-empty string.")
    return dict(filters)


@dataclass(frozen=True)
class VectorSearchResult:
    chunk: KnowledgeChunk
    score: float


class VectorStore(ABC):
    @abstractmethod
    def add(self, chunks: Sequence[KnowledgeChunk], vectors: Sequence[Sequence[float]]) -> list[str]:
        """Upsert corresponding vectors and payloads; return original chunk IDs."""

    @abstractmethod
    def search(
        self, vector: Sequence[float], top_k: int = 5,
        filters: Mapping[str, str] | None = None,
    ) -> list[VectorSearchResult]:
        """Nearest vectors with metadata constraints applied before Top-K."""

    @abstractmethod
    def delete(self, chunk_ids: Sequence[str]) -> None:
        """Delete explicitly named chunks. Repeated deletion is safe."""

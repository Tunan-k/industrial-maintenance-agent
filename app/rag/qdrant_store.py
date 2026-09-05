"""Qdrant local dense vector store with lossless KnowledgeChunk payloads."""
from __future__ import annotations

import math
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Sequence

from qdrant_client import QdrantClient, models

from app.rag.schemas import KnowledgeChunk
from app.rag.vector_store import (
    VectorSearchResult, VectorStore, VectorStoreError, validate_metadata_filters,
)


class QdrantVectorStore(VectorStore):
    def __init__(
        self, *, collection_name: str, vector_size: int,
        path: str | Path = ":memory:", embedding_info: dict | None = None,
    ):
        if not collection_name.strip() or vector_size < 1:
            raise ValueError("collection_name and positive vector_size are required.")
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.embedding_info = dict(embedding_info or {})
        self._client = None
        try:
            self._client = (QdrantClient(location=":memory:") if str(path) == ":memory:"
                            else QdrantClient(path=str(path)))
            if self._client.collection_exists(collection_name):
                params = self._client.get_collection(collection_name).config.params.vectors
                if not isinstance(params, models.VectorParams) or (
                    params.size != vector_size or params.distance != models.Distance.COSINE
                ):
                    raise VectorStoreError("Existing collection dimension or distance is incompatible.")
            else:
                self._client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
                )
        except Exception as exc:
            self.close()
            if isinstance(exc, VectorStoreError):
                raise
            raise VectorStoreError("Unable to initialize Qdrant collection.") from exc

    @staticmethod
    def _point_id(chunk_id: str) -> str:
        # Qdrant accepts UUIDs, not arbitrary chk_* strings. Retain the original
        # ID in the payload and map it deterministically for idempotent upserts.
        return str(uuid.uuid5(uuid.NAMESPACE_URL, "industrial-maintenance/chunk/" + chunk_id))

    def _vector(self, vector: Sequence[float]) -> list[float]:
        values = [float(value) for value in vector]
        if len(values) != self.vector_size:
            raise ValueError("Vector dimension does not match the collection.")
        if not all(math.isfinite(value) for value in values) or not any(values):
            raise ValueError("Vector must be finite and nonzero.")
        # Normalize explicitly so persisted local vectors have the same scale
        # as the vectors used for cosine search, including after reopening.
        scale = max(abs(value) for value in values)
        scaled = [value / scale for value in values]
        norm = math.hypot(*scaled)
        return [value / norm for value in scaled]

    def add(self, chunks: Sequence[KnowledgeChunk], vectors: Sequence[Sequence[float]]) -> list[str]:
        if len(chunks) != len(vectors):
            raise ValueError("A vector is required for every chunk.")
        # Validate the entire batch before making any database write.
        points = [models.PointStruct(
            id=self._point_id(chunk.chunk_id), vector=self._vector(vector),
            payload={**chunk.model_dump(mode="json"), "embedding": self.embedding_info},
        ) for chunk, vector in zip(chunks, vectors)]
        if points:
            try:
                self._client.upsert(self.collection_name, points=points, wait=True)
            except Exception as exc:
                raise VectorStoreError("Qdrant vector write failed.") from exc
        return [chunk.chunk_id for chunk in chunks]

    def search(
        self, vector: Sequence[float], top_k: int = 5,
        filters: Mapping[str, str] | None = None,
    ) -> list[VectorSearchResult]:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be positive.")
        constraints = validate_metadata_filters(filters)
        query_filter = models.Filter(must=[
            models.FieldCondition(key=f"metadata.{key}", match=models.MatchValue(value=value))
            for key, value in constraints.items()
        ]) if constraints else None
        query = self._vector(vector)
        try:
            hits = self._client.query_points(
                collection_name=self.collection_name, query=query,
                limit=top_k, with_payload=True, query_filter=query_filter,
            ).points
            results = []
            for hit in hits:
                payload = hit.payload or {}
                chunk = KnowledgeChunk.model_validate({
                    key: payload[key] for key in ("chunk_id", "document_id", "text", "metadata")
                })
                results.append(VectorSearchResult(chunk=chunk, score=float(hit.score)))
            return results
        except Exception as exc:
            raise VectorStoreError("Qdrant search or chunk payload validation failed.") from exc

    def delete(self, chunk_ids: Sequence[str]) -> None:
        if not chunk_ids:
            return
        try:
            self._client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(points=[self._point_id(id_) for id_ in chunk_ids]),
                wait=True,
            )
        except Exception as exc:
            raise VectorStoreError("Qdrant delete failed.") from exc

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

"""Dense query embedding -> filtered vector search -> source Evidence."""
from collections.abc import Mapping

from app.rag.embedding import EmbeddingProvider
from app.rag.schemas import Evidence
from app.rag.vector_store import VectorStore, validate_metadata_filters


class Retriever:
    def __init__(self, embedding_provider: EmbeddingProvider, vector_store: VectorStore):
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store

    def retrieve(
        self, query: str, top_k: int = 5,
        filters: Mapping[str, str] | None = None,
    ) -> list[Evidence]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string.")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive integer.")
        constraints = validate_metadata_filters(filters)
        vector = self.embedding_provider.embed_text(query)
        hits = self.vector_store.search(vector, top_k=top_k, filters=constraints)
        # Errors intentionally propagate: failures must not masquerade as no hits.
        return [Evidence(
            chunk_id=hit.chunk.chunk_id,
            text=hit.chunk.text,
            score=hit.score,
            metadata=hit.chunk.metadata.model_copy(deep=True),
        ) for hit in hits]

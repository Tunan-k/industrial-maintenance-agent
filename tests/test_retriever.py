"""Offline retrieval contract tests against real chunks and real local Qdrant."""
import hashlib
import math
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.rag.embedding import EmbeddingError, EmbeddingProvider, chunk_embedding_text
from app.rag.qdrant_store import QdrantVectorStore
from app.rag.retriever import Retriever
from app.rag.schemas import Evidence, KnowledgeChunk
from app.rag.vector_store import VectorStoreError


class DeterministicTestEmbedding(EmbeddingProvider):
    """Test-only hash vectors: deterministic plumbing, no semantic capability."""

    dimension = 32
    info = {"provider": "test_hash_only", "dimension": 32}

    def embed_text(self, text):
        values = [byte - 127.5 for byte in hashlib.sha256(text.encode()).digest()]
        norm = math.hypot(*values)
        return [value / norm for value in values]

    def embed_chunks(self, chunks):
        return [self.embed_text(chunk_embedding_text(chunk)) for chunk in chunks]


@pytest.fixture
def chunks():
    path = Path(__file__).resolve().parents[1] / "knowledge/processed/chunks/knowledge_chunks.jsonl"
    return [KnowledgeChunk.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_real_chunks_retrieved_with_scores_and_metadata(tmp_path, chunks):
    provider = DeterministicTestEmbedding()
    with QdrantVectorStore(collection_name="pilot", vector_size=32, path=tmp_path) as store:
        store.add(chunks, provider.embed_chunks(chunks))
        retriever = Retriever(provider, store)
        for target in (chunks[0], chunks[7], chunks[-1]):
            evidence = retriever.retrieve(chunk_embedding_text(target), top_k=3)
            assert len(evidence) == 3
            assert evidence[0].chunk_id == target.chunk_id
            assert evidence[0].text == target.text
            assert evidence[0].metadata == target.metadata
            assert evidence[0].score == pytest.approx(1., abs=1e-5)
            assert all(math.isfinite(item.score) and -1.000001 <= item.score <= 1.000001 for item in evidence)
            assert [item.score for item in evidence] == sorted((item.score for item in evidence), reverse=True)
            assert Evidence.model_validate_json(evidence[0].model_dump_json()) == evidence[0]
            assert set(evidence[0].model_dump()) == {"chunk_id", "text", "score", "metadata"}


def test_equipment_filter_applies_before_top_k(tmp_path, chunks):
    class FixedQueryEmbedding(DeterministicTestEmbedding):
        dimension = 3

        def embed_text(self, text):
            return [1., 0., 0.]

    # The wrong equipment is deliberately closer than the matching one. A
    # post-Top-K Python filter would incorrectly return no result at top_k=1.
    wrong = chunks[0].model_copy(deep=True)
    wrong.chunk_id = "test_compressor"
    wrong.metadata.equipment_type = "compressor"
    matching = chunks[1]
    with QdrantVectorStore(collection_name="mixed", vector_size=3, path=tmp_path) as store:
        store.add([wrong, matching], [[1., 0., 0.], [0.8, 0.6, 0.]])
        retriever = Retriever(FixedQueryEmbedding(), store)
        assert retriever.retrieve("inspect pump", top_k=1)[0].chunk_id == wrong.chunk_id
        hits = retriever.retrieve("inspect pump", top_k=1, filters={"equipment_type": "drilling_pump"})
        assert len(hits) == 1
        assert hits[0].chunk_id == matching.chunk_id
        assert hits[0].metadata.equipment_type == "drilling_pump"
        assert hits[0].score == pytest.approx(0.8, abs=1e-5)
        assert retriever.retrieve("inspect pump", filters={"equipment_type": "unknown_equipment"}) == []


def test_empty_collection_returns_empty_list():
    with QdrantVectorStore(collection_name="empty", vector_size=32) as store:
        assert Retriever(DeterministicTestEmbedding(), store).retrieve("pump") == []


@pytest.mark.parametrize("kwargs", [
    {"query": ""}, {"query": "  "}, {"query": None},
    {"top_k": 0}, {"top_k": -1}, {"top_k": True}, {"top_k": 1.5},
    {"filters": {"equipment_typo": "drilling_pump"}},
    {"filters": {"equipment_type": ""}},
    {"filters": {"equipment_type": None}}, {"filters": []},
])
def test_invalid_request_fails_before_embedding(kwargs):
    class MustNotEncode(DeterministicTestEmbedding):
        def embed_text(self, text):
            pytest.fail("invalid request reached embedding")

    retriever = Retriever(MustNotEncode(), None)
    with pytest.raises(ValueError):
        retriever.retrieve(**{"query": "pump", **kwargs})


def test_embedding_and_database_errors_are_not_empty_results(monkeypatch):
    provider = DeterministicTestEmbedding()
    with QdrantVectorStore(collection_name="failure", vector_size=32) as store:
        retriever = Retriever(provider, store)
        def database_failure(*args, **kwargs):
            raise VectorStoreError("offline database failure")
        monkeypatch.setattr(store, "search", database_failure)
        with pytest.raises(VectorStoreError):
            retriever.retrieve("pump")
        def embedding_failure(text):
            raise EmbeddingError("offline encoding failure")
        monkeypatch.setattr(provider, "embed_text", embedding_failure)
        with pytest.raises(EmbeddingError):
            retriever.retrieve("pump")


@pytest.mark.parametrize("score", [float("nan"), float("inf"), float("-inf")])
def test_evidence_rejects_nonfinite_scores(chunks, score):
    with pytest.raises(ValidationError):
        Evidence(chunk_id=chunks[0].chunk_id, text=chunks[0].text,
                 metadata=chunks[0].metadata, score=score)

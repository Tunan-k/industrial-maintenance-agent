"""Real local Qdrant tests, including persistence and exact payload recovery."""
import math
from pathlib import Path

import pytest
from qdrant_client import QdrantClient, models

from app.rag.qdrant_store import QdrantVectorStore
from app.rag.schemas import KnowledgeChunk
from app.rag.vector_store import VectorStore, VectorStoreError


@pytest.fixture
def chunks():
    path = Path(__file__).resolve().parents[1] / "knowledge/processed/chunks/knowledge_chunks.jsonl"
    return [KnowledgeChunk.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_persistence_payload_and_idempotent_upsert(tmp_path, chunks):
    chunk = chunks[0]
    path = tmp_path / "qdrant"
    with QdrantVectorStore(collection_name="pilot", vector_size=3, path=path,
                           embedding_info={"model": "test-only", "dimension": 3}) as store:
        assert isinstance(store, VectorStore)
        assert store.add([chunk], [[1., 0., 0.]]) == [chunk.chunk_id]
        store.add([chunk], [[1., 0., 0.]])
        assert len(store.search([1., 0., 0.])) == 1
    with QdrantVectorStore(collection_name="pilot", vector_size=3, path=path) as store:
        hit = store.search([1., 0., 0.], top_k=1)[0]
        assert hit.chunk == chunk
        assert hit.chunk.metadata.model_dump() == chunk.metadata.model_dump()
        assert hit.score == pytest.approx(1.)
        store.delete([chunk.chunk_id])
        store.delete([chunk.chunk_id])
        assert store.search([1., 0., 0.]) == []


def test_all_63_payloads_and_vectors_roundtrip(tmp_path, chunks):
    vectors = [[1., float(i + 1), 1.] for i in range(len(chunks))]
    with QdrantVectorStore(collection_name="all", vector_size=3, path=tmp_path / "db") as store:
        store.add(chunks, vectors)
        hits = store.search(vectors[0], top_k=100)
        assert {hit.chunk.chunk_id: hit.chunk for hit in hits} == {c.chunk_id: c for c in chunks}
    # Inspect actual stored vectors and full payload through Qdrant's own API.
    client = QdrantClient(path=str(tmp_path / "db"))
    try:
        points, _ = client.scroll("all", limit=100, with_vectors=True, with_payload=True)
        assert len(points) == 63
        expected = {chunk.chunk_id: vector for chunk, vector in zip(chunks, vectors)}
        for point in points:
            original = expected[point.payload["chunk_id"]]
            norm = math.sqrt(sum(value * value for value in original))
            assert point.vector == pytest.approx([value / norm for value in original], abs=1e-6)
        assert all(set(point.payload) == {"chunk_id", "document_id", "text", "metadata", "embedding"} for point in points)
    finally:
        client.close()


@pytest.mark.parametrize("bad_vector", [[1., 0.], [float("nan"), 0., 1.], [0., 0., 0.]])
def test_invalid_batch_does_not_partially_write(chunks, bad_vector):
    with QdrantVectorStore(collection_name="invalid", vector_size=3) as store:
        with pytest.raises(ValueError):
            store.add(chunks[:2], [[1., 0., 0.], bad_vector])
        assert store.search([1., 0., 0.]) == []
        with pytest.raises(ValueError):
            store.search(bad_vector)


def test_count_mismatch_and_empty_operations(chunks):
    with QdrantVectorStore(collection_name="empty", vector_size=3) as store:
        with pytest.raises(ValueError):
            store.add(chunks[:1], [])
        assert store.add([], []) == []
        store.delete([])
        with pytest.raises(ValueError):
            store.search([1., 0., 0.], top_k=0)


def test_existing_collection_dimension_is_not_overwritten(tmp_path, chunks):
    with QdrantVectorStore(collection_name="existing", vector_size=3, path=tmp_path) as store:
        store.add(chunks[:1], [[1., 0., 0.]])
    with pytest.raises(VectorStoreError, match="incompatible"):
        QdrantVectorStore(collection_name="existing", vector_size=4, path=tmp_path)
    with QdrantVectorStore(collection_name="existing", vector_size=3, path=tmp_path) as store:
        assert store.search([1., 0., 0.])[0].chunk == chunks[0]


def test_wrong_distance_is_rejected(tmp_path):
    client = QdrantClient(path=str(tmp_path))
    try:
        client.create_collection("dot", vectors_config=models.VectorParams(size=3, distance=models.Distance.DOT))
    finally:
        client.close()
    with pytest.raises(VectorStoreError, match="incompatible"):
        QdrantVectorStore(collection_name="dot", vector_size=3, path=tmp_path)

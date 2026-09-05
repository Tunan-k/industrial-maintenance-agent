import json
import os
from pathlib import Path
from unittest.mock import Mock

import pytest

from app.rag.embedding import EmbeddingConfig, EmbeddingError, SentenceTransformerEmbeddingProvider
from app.rag.qdrant_store import QdrantVectorStore
from app.rag.retriever import Retriever
from app.rag.schemas import Evidence, KnowledgeChunk
from app.rag.vector_store import VectorStoreError
from app.tools.knowledge_tool import (
    KnowledgeTool, KnowledgeToolError, KnowledgeToolInput,
    configure_knowledge_tool, retrieve_knowledge,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def clear_configuration():
    configure_knowledge_tool(None)
    yield
    configure_knowledge_tool(None)


def test_delegation_preserves_evidence_and_defaults():
    chunk = KnowledgeChunk.model_validate_json((ROOT / "knowledge/processed/chunks/knowledge_chunks.jsonl")
                                               .read_text(encoding="utf-8").splitlines()[0])
    evidence = Evidence(chunk_id=chunk.chunk_id, text=chunk.text, score=0.8, metadata=chunk.metadata)
    retriever = Mock(spec=Retriever)
    retriever.retrieve.return_value = [evidence]
    configure_knowledge_tool(retriever)
    result = retrieve_knowledge("inspect pump")
    retriever.retrieve.assert_called_once_with("inspect pump", top_k=5, filters=None)
    assert result == [evidence]
    assert set(result[0].model_dump()) == {"chunk_id", "text", "score", "metadata"}
    assert Evidence.model_validate_json(result[0].model_dump_json()) == evidence


def test_equipment_filter_and_empty_result():
    retriever = Mock(spec=Retriever)
    retriever.retrieve.return_value = []
    assert KnowledgeTool(retriever).retrieve_knowledge("inspect pump", "compressor", 3) == []
    retriever.retrieve.assert_called_once_with("inspect pump", top_k=3, filters={"equipment_type": "compressor"})


@pytest.mark.parametrize("kwargs", [
    {"query": " "}, {"query": None}, {"query": 123},
    {"query": "pump", "top_k": 0}, {"query": "pump", "top_k": True},
    {"query": "pump", "top_k": "3"}, {"query": "pump", "equipment_type": " "},
    {"query": "pump", "equipment_type": 1},
])
def test_invalid_input_fails_before_retrieval(kwargs):
    retriever = Mock(spec=Retriever)
    with pytest.raises(KnowledgeToolError) as caught:
        KnowledgeTool(retriever).retrieve_knowledge(**kwargs)
    assert caught.value.code == "invalid_input"
    retriever.retrieve.assert_not_called()


@pytest.mark.parametrize("error,code", [
    (EmbeddingError("private model path"), "embedding_error"),
    (VectorStoreError("private endpoint"), "vector_store_error"),
    (RuntimeError("internal details"), "retrieval_error"),
])
def test_failures_are_distinct_from_empty_results(error, code):
    retriever = Mock(spec=Retriever)
    retriever.retrieve.side_effect = error
    with pytest.raises(KnowledgeToolError) as caught:
        KnowledgeTool(retriever).retrieve_knowledge("pump")
    assert caught.value.code == code
    assert caught.value.__cause__ is error
    assert str(error) not in str(caught.value)


def test_unconfigured_function_has_explicit_error():
    with pytest.raises(KnowledgeToolError) as caught:
        retrieve_knowledge("pump")
    assert caught.value.code == "not_configured"
    assert KnowledgeToolInput(query="pump").top_k == 5


@pytest.mark.skipif(os.environ.get("RUN_SEMANTIC_TESTS") != "1", reason="Opt-in existing real local index")
def test_real_query_through_public_tool():
    # Open the Phase 3.7-B persisted collection. Do not ingest or re-embed chunks.
    path = Path(os.environ.get("KNOWLEDGE_QDRANT_PATH", ROOT / "knowledge/vector_store/phase3_7b"))
    assert (path / "meta.json").is_file(), "Prepare the existing Pilot Qdrant index first."
    provider = SentenceTransformerEmbeddingProvider(EmbeddingConfig.from_env())
    with QdrantVectorStore(collection_name="semantic_validation", vector_size=provider.dimension,
                           path=path, embedding_info=provider.info) as store:
        configure_knowledge_tool(Retriever(provider, store))
        hits = retrieve_knowledge("钻井泵阀盒部署哪些传感器，采样频率是多少？", "drilling_pump", 3)
        originals = {c["chunk_id"]: c for c in [json.loads(line) for line in
                     (ROOT / "knowledge/processed/chunks/knowledge_chunks.jsonl").read_text(encoding="utf-8").splitlines()]}
        assert len(hits) == 3
        for hit in hits:
            assert hit.metadata.equipment_type == "drilling_pump"
            assert hit.text == originals[hit.chunk_id]["text"]
            assert hit.metadata.model_dump(mode="json") == originals[hit.chunk_id]["metadata"]
        assert any("1kHz" in hit.text for hit in hits)
        assert retrieve_knowledge("压力异常可能原因", "compressor", 3) == []
        configure_knowledge_tool(None)

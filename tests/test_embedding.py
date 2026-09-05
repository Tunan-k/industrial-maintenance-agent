"""Offline adapter tests: fake encoder plus a tiny random local model.

The tiny model verifies real inference plumbing, not semantic retrieval quality.
"""
from pathlib import Path

import numpy as np
import pytest

from app.rag.embedding import (
    EmbeddingConfig, EmbeddingError, SentenceTransformerEmbeddingProvider,
    chunk_embedding_text,
)
from app.rag.schemas import KnowledgeChunk


def pilot_chunk():
    path = Path(__file__).resolve().parents[1] / "knowledge/processed/chunks/knowledge_chunks.jsonl"
    return KnowledgeChunk.model_validate_json(path.read_text(encoding="utf-8").splitlines()[0])


class FakeEncoder:
    max_seq_length = 8192

    def __init__(self):
        self.calls = []
        self.output = None

    def get_embedding_dimension(self):
        return 3

    def tokenizer(self, texts, **kwargs):
        return {"input_ids": [[1] * len(text) for text in texts]}

    def encode(self, texts, **kwargs):
        self.calls.append((texts, kwargs))
        return self.output if self.output is not None else np.array([[3., 4., 0.]] * len(texts))


@pytest.fixture
def provider(monkeypatch):
    encoder = FakeEncoder()
    monkeypatch.setattr("app.rag.embedding._load_model", lambda config: encoder)
    result = SentenceTransformerEmbeddingProvider(EmbeddingConfig(
        "test-only", query_prefix="query: ", document_prefix="passage: ",
    ))
    return result, encoder


def test_chunk_embedding_and_context(provider):
    adapter, encoder = provider
    chunk = pilot_chunk()
    before = chunk.model_dump_json()
    vectors = adapter.embed_chunks([chunk, chunk])
    assert np.asarray(vectors).shape == (2, 3)
    assert np.linalg.norm(vectors, axis=1) == pytest.approx([1., 1.])
    assert encoder.calls[0][0] == ["passage: " + chunk_embedding_text(chunk)] * 2
    assert chunk.model_dump_json() == before
    assert adapter.embed_text("检查阀门 / inspect valve") == pytest.approx(vectors[0])
    assert encoder.calls[-1][0] == ["query: 检查阀门 / inspect valve"]
    assert adapter.dimension == adapter.info["dimension"] == 3
    assert adapter.info["normalize"] is True


def test_empty_input(provider):
    adapter, encoder = provider
    assert adapter.embed_chunks([]) == []
    assert not encoder.calls
    with pytest.raises(ValueError):
        adapter.embed_text(" ")


@pytest.mark.parametrize("output", [[[0., 0., 0.]], [[float("nan"), 1., 2.]], [[1., 2.]]])
def test_invalid_encoder_output(provider, output):
    adapter, encoder = provider
    encoder.output = output
    with pytest.raises(EmbeddingError):
        adapter.embed_text("valve")


def test_overlong_input_is_not_silently_truncated(provider):
    adapter, encoder = provider
    encoder.max_seq_length = 4
    with pytest.raises(EmbeddingError, match="token limit"):
        adapter.embed_text("a long input")
    assert not encoder.calls


def test_config_from_environment(monkeypatch):
    monkeypatch.delenv("EMBEDDING_MODEL_PATH", raising=False)
    monkeypatch.setenv("EMBEDDING_MODEL", "some/local/model")
    monkeypatch.setenv("EMBEDDING_REVISION", "revision-a")
    monkeypatch.setenv("EMBEDDING_BATCH_SIZE", "2")
    config = EmbeddingConfig.from_env()
    assert config.model_name_or_path == "some/local/model"
    assert config.revision == "revision-a"
    assert config.batch_size == 2
    assert config.local_files_only is True
    monkeypatch.delenv("EMBEDDING_MODEL")
    with pytest.raises(ValueError):
        EmbeddingConfig.from_env()


def test_model_path_environment_takes_precedence(monkeypatch):
    monkeypatch.setenv("EMBEDDING_MODEL_PATH", "local/trained-model")
    monkeypatch.setenv("EMBEDDING_MODEL", "legacy-model")
    assert EmbeddingConfig.from_env().model_name_or_path == "local/trained-model"


def test_load_failure_is_explicit(monkeypatch):
    def fail(config):
        raise OSError("test failure")
    monkeypatch.setattr("app.rag.embedding._load_model", fail)
    with pytest.raises(EmbeddingError, match="load"):
        SentenceTransformerEmbeddingProvider(EmbeddingConfig("missing-model"))


def test_real_local_encoder_to_qdrant(tmp_path, monkeypatch):
    # Construct an untrained, tiny model entirely offline to exercise the real
    # SentenceTransformer loader/encode and Qdrant, without external weights.
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    from transformers import BertConfig, BertModel, BertTokenizer
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.sentence_transformer import modules as models
    from app.rag.qdrant_store import QdrantVectorStore

    model_dir = tmp_path / "tiny-bert"
    model_dir.mkdir()
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "inspect", "valve", "检", "查", "阀", "门"]
    vocab_path = model_dir / "vocab.txt"
    vocab_path.write_text("\n".join(vocab), encoding="utf-8")
    tokenizer = BertTokenizer(vocab_file=str(vocab_path))
    tokenizer.save_pretrained(model_dir)
    BertModel(BertConfig(
        vocab_size=len(tokenizer), hidden_size=16, num_hidden_layers=1,
        num_attention_heads=2, intermediate_size=32, max_position_embeddings=512,
    )).save_pretrained(model_dir)
    transformer = models.Transformer(str(model_dir), max_seq_length=512)
    saved = tmp_path / "sentence-model"
    SentenceTransformer(modules=[transformer, models.Pooling(16)]).save(str(saved))
    adapter = SentenceTransformerEmbeddingProvider(EmbeddingConfig(str(saved)))
    chunk_data = pilot_chunk().model_dump(mode="json")
    chunk_data["text"] = "检查阀门 inspect valve"
    chunk_data["metadata"]["title"] = "valve"
    chunk_data["metadata"]["section"] = None
    chunk = KnowledgeChunk.model_validate(chunk_data)
    vector = adapter.embed_chunks([chunk])[0]
    assert len(vector) == 16
    assert np.linalg.norm(vector) == pytest.approx(1., abs=1e-5)
    assert adapter.embed_chunks([chunk])[0] == pytest.approx(vector, abs=1e-6)
    with QdrantVectorStore(collection_name="tiny", vector_size=16,
                           path=tmp_path / "db", embedding_info=adapter.info) as store:
        assert store.add([chunk], [vector]) == [chunk.chunk_id]
        hit = store.search(vector, top_k=1)[0]
        assert hit.chunk == chunk
        assert hit.score == pytest.approx(1., abs=1e-5)

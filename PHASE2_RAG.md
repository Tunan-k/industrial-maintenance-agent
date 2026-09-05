# Phase 2: Embedding and local Qdrant

This phase adds storage primitives only. It does not connect diagnosis APIs,
implement a Retriever, or call an Agent/Tool.

## Environment

Use the existing Python 3.11 project venv. Additive installation:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements-rag-phase2.txt
```

The phase was tested with qdrant-client 1.19.0 and sentence-transformers 6.0.1.
The pre-existing application requirements files are not replaced by this file.

## Embedding contract

- `EmbeddingProvider.dimension` and `.info`: dimension, model identity and config.
- `embed_text(text)`: one query vector.
- `embed_chunks(chunks)`: ordered vectors for title + section + original text.
- `EmbeddingConfig`: replaceable local path/model ID, revision, device, batch size,
  normalization, query/document prefixes. Choose a trained Chinese/English model.
- `EmbeddingConfig.from_env()` reads exported variables from `.env.example`.
  It does not load a `.env` file. Local-only loading is the default.
- No pretrained weights are bundled or downloaded automatically. Missing models
  raise `EmbeddingError`; fake vectors are never a production fallback.
- Follow the chosen model card for prefixes. Query and document must use the same
  compatible model/revision. Inputs exceeding its token limit fail explicitly;
  this phase does not change the 63 chunks or silently truncate them.

## One chunk into Qdrant

Run from the repository root, after exporting `EMBEDDING_MODEL` pointing to an
existing trained local Sentence Transformers model. Optional variables are shown
in `.env.example`. This is a live-model example; the pretrained model must exist.

```python
import os
from pathlib import Path

from app.rag.embedding import EmbeddingConfig, SentenceTransformerEmbeddingProvider
from app.rag.qdrant_store import QdrantVectorStore
from app.rag.schemas import KnowledgeChunk

lines = Path("knowledge/processed/chunks/knowledge_chunks.jsonl").read_text(
    encoding="utf-8"
).splitlines()
chunk = KnowledgeChunk.model_validate_json(lines[0])
provider = SentenceTransformerEmbeddingProvider(EmbeddingConfig.from_env())
vector = provider.embed_chunks([chunk])[0]

with QdrantVectorStore(
    collection_name=os.environ.get("QDRANT_COLLECTION", "maintenance_v1"),
    vector_size=provider.dimension,
    path=os.environ.get("QDRANT_PATH", "knowledge/vector_store"),
    embedding_info=provider.info,
) as store:
    store.add([chunk], [vector])
    hit = store.search(vector, top_k=1)[0]
    assert hit.chunk == chunk
    print(hit.chunk.chunk_id, hit.score, hit.chunk.metadata.model_dump())
    # Explicit deletion, when desired:
    # store.delete([chunk.chunk_id])
```

`search()` accepts a vector and returns `VectorSearchResult(chunk, score)`. It
does not embed text, select source scopes, or apply Retriever business policy.

The store uses a single dense cosine collection. Default `path=":memory:"` is
ephemeral; pass a filesystem path for persistence and close it before reopening.
Original IDs are mapped to stable UUIDs for Qdrant upsert; full original IDs,
text, typed metadata and embedding config are retained in the point payload.
Repeat `add()` for the same chunk ID updates that point. `delete()` removes only
the specified IDs. Rebuilding a changed document must explicitly remove its old
chunk IDs; automatic document-version reconciliation is outside this phase.

Existing collection dimensions/distance are checked and never silently replaced.
Different models of the same dimension are not automatically detected: use a new
collection or explicitly rebuild when changing embedding model/configuration.

## Offline validation

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
.\venv\Scripts\python.exe -B -m pytest -p no:cacheprovider -q tests/test_embedding.py tests/test_vector_store.py
.\venv\Scripts\python.exe -B -m pytest -p no:cacheprovider -q
```

Unit tests use a fake encoder. An additional offline integration test constructs
a tiny **random, untrained** local BERT/SentenceTransformer and runs its real
inference through the provider into real persisted Qdrant. This verifies the
pipeline, not semantic retrieval quality. No pretrained-model quality claim is
made. Vector-store tests round-trip all 63 existing chunk payloads, check stored
cosine-normalized vectors, close/reopen persistence, deletion, repeat upserts,
invalid vectors, and collection incompatibility.

API references used:
- [Sentence Transformers API](https://www.sbert.net/docs/package_reference/sentence_transformer/model.html)
- [Qdrant client API](https://github.com/qdrant/qdrant-client)

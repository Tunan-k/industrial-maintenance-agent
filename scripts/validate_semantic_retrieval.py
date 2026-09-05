"""Opt-in real local embedding validation over the tracked industrial corpus."""
from __future__ import annotations

import argparse
import json
import os
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

from app.rag.embedding import EmbeddingConfig, EmbeddingError, SentenceTransformerEmbeddingProvider
from app.rag.qdrant_store import QdrantVectorStore
from app.rag.retriever import Retriever
from app.rag.schemas import KnowledgeChunk


ROOT = Path(__file__).resolve().parents[1]
QUERIES = [
    "钻井泵吸入阀故障原因和检查方法",
    "压力异常可能原因",
    "设备振动异常处理措施",
]


def validate_semantic_retrieval(top_k: int = 3, queries: list[str] | None = None,
                                qdrant_path: Path | None = None) -> dict:
    model_path = os.environ.get("EMBEDDING_MODEL_PATH", "")
    if not model_path or not Path(model_path).is_dir():
        raise ValueError("Set EMBEDDING_MODEL_PATH to an existing trained local model directory.")
    provider = SentenceTransformerEmbeddingProvider(EmbeddingConfig.from_env())
    chunks = [KnowledgeChunk.model_validate_json(line) for line in (
        ROOT / "knowledge/processed/chunks/knowledge_chunks.jsonl"
    ).read_text(encoding="utf-8").splitlines() if line.strip()]
    included, vectors, excluded = [], [], []
    started = perf_counter()
    def embed_batch(batch):
        try:
            batch_vectors = provider.embed_chunks(batch)
        except EmbeddingError as exc:
            # Preserve original chunks. Report model-limit exclusions instead of
            # silently truncating evidence or inventing substitute embeddings.
            if str(exc) != "Input exceeds the configured model token limit.":
                raise
            if len(batch) > 1:
                middle = len(batch) // 2
                embed_batch(batch[:middle])
                embed_batch(batch[middle:])
            else:
                excluded.append({"chunk_id": batch[0].chunk_id, "reason": str(exc)})
            return
        included.extend(batch)
        vectors.extend(batch_vectors)

    for start in range(0, len(chunks), provider.config.batch_size):
        embed_batch(chunks[start:start + provider.config.batch_size])
    if not included:
        raise RuntimeError("No original chunk fits the configured embedding model.")
    if qdrant_path is not None:
        qdrant_path.mkdir(parents=True, exist_ok=True)
    with (nullcontext(str(qdrant_path)) if qdrant_path is not None
          else TemporaryDirectory(prefix="semantic_retrieval_")) as directory:
        with QdrantVectorStore(collection_name="semantic_validation",
                               vector_size=provider.dimension, path=directory,
                               embedding_info=provider.info) as store:
            store.add(included, vectors)
            retriever = Retriever(provider, store)
            results = []
            for query in QUERIES if queries is None else queries:
                begin = perf_counter()
                evidence = retriever.retrieve(query, top_k=top_k,
                                               filters={"equipment_type": "drilling_pump"})
                results.append({"query": query, "latency_ms": round((perf_counter()-begin)*1000, 2),
                                "evidence": [item.model_dump(mode="json") for item in evidence]})
            no_match = retriever.retrieve(QUERIES[0], top_k=top_k,
                                          filters={"equipment_type": "compressor"})
    return {
        "embedding": provider.info, "top_k": top_k,
        "input_chunks": len(chunks), "indexed_chunks": len(included),
        "excluded_chunks": excluded, "results": results,
        "nonmatching_equipment_results": [item.model_dump(mode="json") for item in no_match],
        "total_latency_ms": round((perf_counter()-started)*1000, 2),
        "notes": ["Scores are cosine similarity, not confidence or fault probability.",
                  "Topic matching does not prove that evidence contains actionable maintenance guidance."],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = validate_semantic_retrieval(args.top_k)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

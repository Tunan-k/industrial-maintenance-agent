"""Idempotent metadata-only pilot migration. No chunk regeneration or text changes."""
from copy import deepcopy
import json

from app.rag.schemas import KnowledgeChunk, KnowledgeDocument
from app.rag.source_metadata import resolve_source_type
from scripts.evaluate_retrieval import ROOT, CORPUS, DATASET, corpus_digest


def repair_chunks(chunks):
    result = deepcopy(chunks)
    for chunk in result:
        metadata = chunk["metadata"]
        metadata["source_type"] = resolve_source_type(metadata.get("document_type"), metadata["source_format"]).value
        KnowledgeChunk.model_validate(chunk)
    return result


def main():
    before = [json.loads(line) for line in CORPUS.read_text(encoding="utf-8").splitlines()]
    dataset = json.loads(DATASET.read_text(encoding="utf-8"))
    if dataset["corpus_sha256"] != corpus_digest(before):
        raise ValueError("Review corpus drift before migrating the evaluation fingerprint.")
    after = repair_chunks(before)
    documents = []
    for path in (ROOT / "knowledge/processed/documents").glob("*.knowledge.json"):
        document = json.loads(path.read_text(encoding="utf-8"))
        document["source_type"] = resolve_source_type(document["metadata"].get("document_type"), document["source_format"]).value
        KnowledgeDocument.model_validate(document)
        documents.append((path, document))
    # Explicitly prove that only source_type changed before migrating the corpus hash.
    for old, new in zip(before, after):
        restored = deepcopy(new)
        restored["metadata"]["source_type"] = old["metadata"]["source_type"]
        assert restored == old
    dataset["corpus_sha256"] = corpus_digest(after)
    dataset["version"] = "phase3.7b-metadata-only"
    dataset["migration_note"] = "Only corpus source_type changed. Queries, expectations and relevance labels remain frozen for comparison."
    # Preserve the original expectations as well as relevance labels; semantic
    # source_type matching is not part of Top1 or Precision@3 scoring.
    CORPUS.write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in after) + "\n", encoding="utf-8")
    for path, document in documents:
        path.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    DATASET.write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Validated {len(after)} chunks; updated only source_type; retained all evaluation cases.")


if __name__ == "__main__":
    main()

"""Contract and offline regression tests; no parser/model downloads required."""
import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.rag.chunking import (
    ChunkingConfig, KnowledgeChunk as ChunkingChunk, chunk_document,
    load_active_documents, save_chunks, stable_chunk_id,
)
from app.rag.schemas import KnowledgeChunk, KnowledgeDocument

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_METADATA = {
    "title", "source_type", "source_uri", "equipment_type",
    "authority_level", "page", "section",
}
# Fingerprint of the 63 pre-migration (chunk_id, document_id, text) tuples.
BASELINE_SHA256 = "07fd8c7dd63679a1e0ec630df6162a28a213e07c01c77998a9f59f286cbc55e7"


def document_for_format(tmp_path, source_format):
    document = KnowledgeDocument(
        document_id="doc_fixture", title="Pump inspection",
        source_name=f"manual.{source_format}", source_format=source_format,
        source_type="maintenance_sop", source_uri=f"manuals/manual.{source_format}",
        text="## Inspection\n\nInspect the suction valve and record wear.",
        equipment_type="drilling_pump", equipment_model="HH2400",
        authority_level=4, language="en",
        metadata={"authority_label": "internal_reference", "manifest_document_id": "src_1"},
    ).model_dump(mode="json")
    if source_format == "pdf":
        pages = tmp_path / "pages.json"
        pages.write_text(json.dumps({"pages": [{"page_number": 2, "text": "Inspect the suction valve and record wear."}]}), encoding="utf-8")
        document["metadata"]["pdf_pages_path"] = str(pages)
    return document


@pytest.mark.parametrize("source_format", ["html", "pdf", "pptx"])
def test_unified_format_contract(tmp_path, source_format):
    document = document_for_format(tmp_path, source_format)
    chunks = chunk_document(document, ChunkingConfig())
    assert chunks
    assert ChunkingChunk is KnowledgeChunk
    for chunk in chunks:
        data = chunk.model_dump(mode="json")
        assert set(data) == {"chunk_id", "document_id", "text", "metadata"}
        assert REQUIRED_METADATA <= data["metadata"].keys()
        assert chunk.metadata.title == document["title"]
        for key in ("source_type", "source_uri", "equipment_type", "equipment_model", "authority_level"):
            assert data["metadata"][key] == document[key]
        assert chunk.metadata.authority_label == "internal_reference"
        assert chunk.metadata.page == (2 if source_format == "pdf" else None)
        assert chunk.metadata.section == (None if source_format == "pdf" else "Inspection")
        assert KnowledgeChunk.model_validate_json(chunk.model_dump_json()) == chunk
    assert chunks == chunk_document(document, ChunkingConfig())
    output = tmp_path / "chunks.jsonl"
    save_chunks(chunks, output)
    assert [KnowledgeChunk.model_validate_json(line) for line in output.read_text(encoding="utf-8").splitlines()] == chunks


@pytest.mark.parametrize("field", sorted(REQUIRED_METADATA))
def test_required_metadata_cannot_be_omitted(tmp_path, field):
    data = chunk_document(document_for_format(tmp_path, "html"), ChunkingConfig())[0].model_dump(mode="json")
    del data["metadata"][field]
    with pytest.raises(ValidationError):
        KnowledgeChunk.model_validate(data)


@pytest.mark.parametrize("field", ["chunk_id", "document_id", "text", "metadata"])
def test_required_chunk_fields(tmp_path, field):
    data = chunk_document(document_for_format(tmp_path, "html"), ChunkingConfig())[0].model_dump(mode="json")
    del data[field]
    with pytest.raises(ValidationError):
        KnowledgeChunk.model_validate(data)


def test_numeric_authority_and_unknown_provenance(tmp_path):
    data = chunk_document(document_for_format(tmp_path, "html"), ChunkingConfig())[0].model_dump(mode="json")
    for key in ("source_uri", "equipment_type", "page", "section"):
        data["metadata"][key] = None
    KnowledgeChunk.model_validate(data)
    for invalid in (0, 6, "industry_association"):
        data["metadata"]["authority_level"] = invalid
        with pytest.raises(ValidationError):
            KnowledgeChunk.model_validate(data)


def test_id_changes_when_content_changes():
    args = dict(document_id="doc_a", chunk_index=0, page=1, section=None, text="Inspect valve.")
    assert stable_chunk_id(**args) == stable_chunk_id(**args)
    assert stable_chunk_id(**args) != stable_chunk_id(**{**args, "text": "Inspect seal."})


def test_real_pilot_preserves_63_chunks(tmp_path):
    documents = load_active_documents(
        manifest_path=ROOT / "knowledge/manifests/knowledge_sources.jsonl",
        document_dir=ROOT / "knowledge/processed/documents",
    )
    # Resolve tracked page artifacts locally, independent of the author's drive.
    for document in documents:
        if document["source_format"] == "pdf":
            document["metadata"]["pdf_pages_path"] = str(ROOT / "knowledge/processed/pdf_pages" / f"{document['document_id']}.pages.json")
    chunks = [chunk for document in documents for chunk in chunk_document(document, ChunkingConfig())]
    projection = [[c.chunk_id, c.document_id, c.text] for c in chunks]
    fingerprint = hashlib.sha256(json.dumps(projection, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    assert len(chunks) == 63
    assert len({c.chunk_id for c in chunks}) == 63
    assert Counter(c.metadata.source_format.value for c in chunks) == {"html": 7, "pdf": 53, "pptx": 3}
    assert fingerprint == BASELINE_SHA256
    assert chunks == [chunk for document in documents for chunk in chunk_document(document, ChunkingConfig())]
    stored = [KnowledgeChunk.model_validate_json(line) for line in (ROOT / "knowledge/processed/chunks/knowledge_chunks.jsonl").read_text(encoding="utf-8").splitlines()]
    assert chunks == stored

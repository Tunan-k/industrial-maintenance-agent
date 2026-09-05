from copy import deepcopy
import pytest
from app.rag.source_metadata import resolve_source_type
from scripts.evaluate_retrieval import load_dataset
from scripts.repair_pilot_metadata import repair_chunks
from app.rag.embedding import chunk_embedding_text
from app.rag.schemas import KnowledgeChunk


@pytest.mark.parametrize("meaning,fmt,expected", [
    ("research_paper", "pdf", "research_paper"),
    ("industry_safety_guidance", "html", "industry_safety_guidance"),
    ("internal_project_note", "pptx", "internal_project_note"),
    (None, ".pptx", "ppt"), (None, "ppt", "ppt"),
    ("oem_manual", "pdf", "oem_manual"), (None, "pdf", "other"),
])
def test_source_mapping(meaning, fmt, expected):
    assert resolve_source_type(meaning, fmt).value == expected


def test_repair_preserves_every_field_except_source_type_and_is_idempotent():
    _, chunks = load_dataset()
    legacy = deepcopy(chunks)
    for c in legacy:
        c["metadata"]["source_type"] = "other"
    repaired = repair_chunks(legacy)
    assert repaired == chunks
    assert repair_chunks(repaired) == repaired
    for old, new in zip(legacy, repaired):
        assert chunk_embedding_text(KnowledgeChunk.model_validate(old)) == chunk_embedding_text(KnowledgeChunk.model_validate(new))
        restored = deepcopy(new)
        restored["metadata"]["source_type"] = "other"
        assert restored == old

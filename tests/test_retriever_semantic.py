"""Opt-in live local semantic checks. Never fall back to synthetic embeddings."""
import json
import math
import os
from pathlib import Path

import pytest

from app.rag.schemas import Evidence
from scripts.validate_semantic_retrieval import ROOT, validate_semantic_retrieval


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_SEMANTIC_TESTS") != "1",
    reason="Set RUN_SEMANTIC_TESTS=1 and EMBEDDING_MODEL_PATH for real-model tests.",
)


@pytest.fixture(scope="module")
def report():
    return validate_semantic_retrieval(top_k=3)


def test_corpus_coverage_and_filter(report):
    assert report["input_chunks"] == 63
    assert report["indexed_chunks"] > 0
    assert report["indexed_chunks"] + len(report["excluded_chunks"]) == 63
    assert report["nonmatching_equipment_results"] == []


@pytest.mark.parametrize("index,topic_terms", [
    (0, ["吸入", "吸阀", "suction", "inlet", "阀门", "valve"]),
    (1, ["压力", "pressure"]),
    (2, ["振动", "vibration"]),
])
def test_top_k_has_traceable_topic_evidence(report, index, topic_terms):
    source = {item["chunk_id"]: item for item in (
        json.loads(line) for line in (ROOT / "knowledge/processed/chunks/knowledge_chunks.jsonl")
        .read_text(encoding="utf-8").splitlines()
    )}
    hits = report["results"][index]["evidence"]
    assert len(hits) == report["top_k"]
    assert len({item["chunk_id"] for item in hits}) == len(hits)
    scores = [item["score"] for item in hits]
    assert scores == sorted(scores, reverse=True)
    for item in hits:
        Evidence.model_validate(item)
        original = source[item["chunk_id"]]
        assert item["text"] == original["text"]
        assert item["metadata"] == original["metadata"]
        assert item["metadata"]["equipment_type"] == "drilling_pump"
        assert item["metadata"]["source_uri"]
        assert item["metadata"]["source_name"]
        if item["metadata"]["source_format"] == "pdf":
            assert item["metadata"]["page"] >= 1
        assert math.isfinite(item["score"])
    # A deliberately modest lexical sanity check on retrieved original text.
    # This is not an assertion that the source proves a cause or repair action.
    assert any(term in item["text"].lower() for item in hits for term in topic_terms)

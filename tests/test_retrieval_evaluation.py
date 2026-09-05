"""Evaluation arithmetic and annotation integrity; quality is reported, not faked."""
from copy import deepcopy
import json
import os

import pytest

from scripts.evaluate_retrieval import (
    CORPUS, DATASET, compute_metrics, load_dataset, render_markdown, score_results,
)
from scripts.validate_semantic_retrieval import validate_semantic_retrieval


def test_metrics_use_fixed_denominators_and_strict_relevance():
    report = compute_metrics([
        {"evidence": [{"grade": 2}, {"grade": 1}, {"grade": 2}, {"grade": 2}]},
        {"evidence": []},
        {"evidence": [{"grade": 0}, {"grade": 2}]},
    ])
    assert report["top1_hit_rate"] == pytest.approx(1 / 3)
    assert report["top3_relevance_rate"] == pytest.approx(3 / 9)
    assert report["hit_at_3"] == pytest.approx(2 / 3)
    assert compute_metrics([])["top1_hit_rate"] is None


def test_dataset_has_complete_frozen_labels_and_translation_controls():
    dataset, chunks = load_dataset()
    assert len(dataset["cases"]) >= 10
    assert len(chunks) == 63
    assert dataset["annotation"]["review_status"] == "pending_human_review"
    assert dataset["annotation"]["human_reviewer"] is None
    for pair in ("lockout", "vmd"):
        cases = [c for c in dataset["cases"] if c["pair_id"] == pair]
        assert {c["language"] for c in cases} == {"zh", "en"}
        assert cases[0]["labels"] == cases[1]["labels"]


def test_corpus_drift_requires_new_label_review(tmp_path):
    corpus = tmp_path / "changed.jsonl"
    corpus.write_text(CORPUS.read_text(encoding="utf-8").replace("Lockout", "Changed", 1), encoding="utf-8")
    with pytest.raises(ValueError, match="Corpus changed"):
        load_dataset(DATASET, corpus)


@pytest.mark.parametrize("mutation", ["missing", "invalid_grade", "unsupported_equipment"])
def test_bad_annotations_fail_instead_of_becoming_misses(tmp_path, mutation):
    dataset, _ = load_dataset()
    case = dataset["cases"][0]
    cid = next(iter(case["labels"]))
    if mutation == "missing":
        del case["labels"][cid]
    elif mutation == "invalid_grade":
        case["labels"][cid]["grade"] = True
    else:
        case["expected_equipment_type"] = "compressor"
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(dataset), encoding="utf-8")
    with pytest.raises(ValueError):
        load_dataset(path)


def test_unknown_or_altered_evidence_is_rejected():
    dataset, chunks = load_dataset()
    original = chunks[0]
    evidence = {key: deepcopy(original[key]) for key in ("chunk_id", "text", "metadata")}
    evidence["score"] = 0.8
    raw = {"results": [{"query": c["query"], "evidence": [deepcopy(evidence)]}
                       for c in dataset["cases"]], "excluded_chunks": []}
    raw["results"][0]["evidence"][0]["text"] = "invented evidence"
    with pytest.raises(ValueError, match="frozen corpus"):
        score_results(dataset, raw, chunks)
    raw["results"][0]["evidence"] = [evidence, evidence]
    with pytest.raises(ValueError, match="Duplicate evidence"):
        score_results(dataset, raw, chunks)


@pytest.mark.skipif(os.environ.get("RUN_SEMANTIC_TESTS") != "1", reason="Opt-in real local embedding evaluation")
def test_real_evaluation_has_traceable_filtered_ranked_results():
    dataset, chunks = load_dataset()
    raw = validate_semantic_retrieval(3, [c["query"] for c in dataset["cases"]])
    report = score_results(dataset, raw, chunks)
    assert report["input_chunks"] == report["indexed_chunks"] + len(report["excluded_chunks"])
    assert report["nonmatching_equipment_results"] == []
    assert report["metrics"]["queries"] >= 10
    assert 0 <= report["metrics"]["top1_hit_rate"] <= 1
    assert 0 <= report["metrics"]["top3_relevance_rate"] <= 1
    for row in report["results"]:
        assert len(row["evidence"]) == 3
        assert all(e["equipment_match"] for e in row["evidence"])
        assert all(e["source"] and "page" in e for e in row["evidence"])
    assert "pending_human_review" == report["annotation"]["review_status"]
    assert "Precision@3" in render_markdown(report)

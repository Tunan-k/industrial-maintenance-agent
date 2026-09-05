"""Reproducible dense retrieval evaluation using explicit, versioned relevance labels."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

from app.rag.schemas import SourceType
from scripts.validate_semantic_retrieval import ROOT, validate_semantic_retrieval

CORPUS = ROOT / "knowledge/processed/chunks/knowledge_chunks.jsonl"
DATASET = ROOT / "tests/fixtures/retrieval_evaluation.json"


def corpus_digest(chunks):
    canonical = json.dumps(chunks, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_dataset(path=DATASET, corpus_path=CORPUS):
    chunks = [json.loads(line) for line in corpus_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    dataset = json.loads(path.read_text(encoding="utf-8"))
    ids = {chunk["chunk_id"] for chunk in chunks}
    if dataset["corpus_sha256"] != corpus_digest(chunks):
        raise ValueError("Corpus changed: review relevance labels before evaluation.")
    cases = dataset["cases"]
    if len(cases) < 10 or len({case["id"] for case in cases}) != len(cases):
        raise ValueError("At least 10 uniquely identified cases are required.")
    for case in cases:
        if not case["query"].strip() or not case["expected_equipment_type"]:
            raise ValueError("Missing query or equipment expectation.")
        if case["expected_equipment_type"] != "drilling_pump":
            raise ValueError("This corpus evaluation uses the drilling_pump filter only.")
        SourceType(case["expected_source_type"])
        if set(case["labels"]) != ids:
            raise ValueError("Every corpus chunk must have an explicit relevance judgment.")
        for label in case["labels"].values():
            if type(label["grade"]) is not int or label["grade"] not in (0, 1, 2) or not label["reason"]:
                raise ValueError("Invalid relevance grade or missing rationale.")
    return dataset, chunks


def compute_metrics(results):
    """Grade 2 is relevant; missing ranks count as misses, never shrink denominators."""
    if not results:
        return {"queries": 0, "top1_hits": 0, "top1_hit_rate": None,
                "top3_relevant": 0, "top3_relevance_rate": None, "hit_at_3": None}
    top1 = sum(bool(r["evidence"]) and r["evidence"][0]["grade"] == 2 for r in results)
    relevant = sum(e["grade"] == 2 for r in results for e in r["evidence"][:3])
    hit3 = sum(any(e["grade"] == 2 for e in r["evidence"][:3]) for r in results)
    return {"queries": len(results), "top1_hits": top1, "top1_hit_rate": top1 / len(results),
            "top3_relevant": relevant, "top3_relevance_rate": relevant / (3 * len(results)),
            "hit_at_3": hit3 / len(results)}


def score_results(dataset, raw, chunks):
    if len(raw["results"]) != len(dataset["cases"]):
        raise ValueError("Query/result count mismatch.")
    originals = {c["chunk_id"]: c for c in chunks}
    excluded = {c["chunk_id"] for c in raw["excluded_chunks"]}
    results = []
    for case, retrieval in zip(dataset["cases"], raw["results"]):
        if case["query"] != retrieval["query"]:
            raise ValueError("Query/result order mismatch.")
        hits = retrieval["evidence"]
        if [e["score"] for e in hits] != sorted((e["score"] for e in hits), reverse=True):
            raise ValueError("Evidence must retain retrieval rank order.")
        if len({e["chunk_id"] for e in hits}) != len(hits):
            raise ValueError("Duplicate evidence would inflate precision.")
        evidence = []
        for rank, item in enumerate(hits, 1):
            original = originals[item["chunk_id"]]
            if item["text"] != original["text"] or item["metadata"] != original["metadata"]:
                raise ValueError("Evidence does not match the frozen corpus.")
            if not math.isfinite(item["score"]):
                raise ValueError("Non-finite score.")
            metadata = item["metadata"]
            evidence.append({**item, "rank": rank, **case["labels"][item["chunk_id"]],
                             "source": metadata["source_name"], "page": metadata["page"],
                             "equipment_match": metadata["equipment_type"] == case["expected_equipment_type"],
                             "source_type_match": metadata["source_type"] == case["expected_source_type"]})
        positive_ids = [cid for cid, label in case["labels"].items() if label["grade"] == 2]
        results.append({key: case[key] for key in ("id", "query", "language", "expected_equipment_type",
                                                  "expected_source_type", "intent", "pair_id")} |
                       {"answerable_in_corpus": bool(positive_ids), "relevant_chunk_ids": positive_ids,
                        "relevant_indexed_chunk_ids": [cid for cid in positive_ids if cid not in excluded],
                        "evidence": evidence})
    selected = [e for r in results for e in r["evidence"][:3]]
    indexed = [c for c in chunks if c["chunk_id"] not in excluded]
    return {
        "annotation": dataset["annotation"], "dataset_version": dataset["version"],
        "corpus_sha256": dataset["corpus_sha256"],
        "dataset_sha256": hashlib.sha256(json.dumps(dataset, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
        "metric_definition": {"relevant": "grade == 2; grade 1 is partial, not an answer",
                              "top1_hit_rate": "relevant rank-1 results / all queries",
                              "top3_relevance_rate": "relevant results at ranks 1-3 / (3 * all queries)",
                              "missing_results": "count as misses", "status": "provisional; labels need human review"},
        "metrics": compute_metrics(results),
        "answerable_metrics": compute_metrics([r for r in results if r["answerable_in_corpus"]]),
        "by_language": {lang: compute_metrics([r for r in results if r["language"] == lang])
                        for lang in sorted({r["language"] for r in results})},
        "diagnostics": {
            "corpus_format_counts": dict(Counter(c["metadata"]["source_format"] for c in chunks)),
            "indexed_format_counts": dict(Counter(c["metadata"]["source_format"] for c in indexed)),
            "top3_format_counts": dict(Counter(e["metadata"]["source_format"] for e in selected)),
            "irrelevant_top3_format_counts": dict(Counter(e["metadata"]["source_format"] for e in selected if e["grade"] == 0)),
            "source_type_counts": dict(Counter(c["metadata"]["source_type"] for c in chunks)),
            "equipment_matches": sum(e["equipment_match"] for e in selected),
            "source_type_matches": sum(e["source_type_match"] for e in selected),
            "returned_top3_slots": len(selected),
            "missing_page_top3": sum(e["page"] is None for e in selected),
            "no_direct_answer_queries": [r["id"] for r in results if not r["answerable_in_corpus"]]},
        "embedding": raw["embedding"], "input_chunks": raw["input_chunks"],
        "indexed_chunks": raw["indexed_chunks"], "excluded_chunks": raw["excluded_chunks"],
        "nonmatching_equipment_results": raw["nonmatching_equipment_results"], "results": results,
    }


def render_markdown(report):
    m = report["metrics"]
    lines = ["# Retriever 质量评估", "",
             "初始标签由 Codex 对照原文审阅，待人工复核；以下是暂定指标，不是人工验收结果。", "",
             f"Top1 命中率：{m['top1_hits']}/{m['queries']} = {m['top1_hit_rate']:.2%}。",
             f"Top3 相关率（Precision@3）：{m['top3_relevant']}/{3*m['queries']} = {m['top3_relevance_rate']:.2%}。",
             "标签：2=正文直接支持问题；1=背景或部分支持；0=不支持。仅 2 计入命中；空结果计未命中。",
             f"索引覆盖：{report['indexed_chunks']}/{report['input_chunks']}；排除项在 JSON 中逐条记录。", ""]
    for r in report["results"]:
        lines.extend([f"## {r['id']}：{r['query']}", "",
                      f"期望 equipment_type={r['expected_equipment_type']}；source_type={r['expected_source_type']}。",
                      f"原始语料存在直接答案：{r['answerable_in_corpus']}。", "",
                      "| Rank | chunk_id | score | source | page | 标签 | 理由 |",
                      "|---|---|---|---|---|---|---|"])
        for e in r["evidence"]:
            lines.append(f"| {e['rank']} | {e['chunk_id']} | {e['score']:.6f} | {e['source']} | {e['page'] if e['page'] is not None else 'null'} | {e['grade']} | {e['reason']} |")
        lines.append("")
    lines.extend(["## 可观测诊断统计", "", "```json",
                  json.dumps(report["diagnostics"], ensure_ascii=False, indent=2), "```", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--qdrant-path", type=Path, help="Persist the rebuilt evaluation collection locally.")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/phase3_6_retrieval.json")
    args = parser.parse_args()
    if args.top_k < 3:
        parser.error("--top-k must be at least 3 to measure Precision@3")
    dataset, chunks = load_dataset(args.dataset)
    raw = validate_semantic_retrieval(args.top_k, queries=[c["query"] for c in dataset["cases"]],
                                      qdrant_path=args.qdrant_path)
    report = score_results(dataset, raw, chunks)
    report["qdrant_path"] = str(args.qdrant_path.resolve()) if args.qdrant_path else None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown = render_markdown(report)
    args.output.with_suffix(".md").write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()

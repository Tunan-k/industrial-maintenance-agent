from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from app.agent.maintenance_agent import MaintenanceAgent
from app.rag.schemas import Evidence, KnowledgeChunk
from app.schemas.maintenance import MaintenanceContext, MaintenanceReport
from app.tools.diagnosis_tool import DiagnosisEvidence, DiagnosisToolError
from app.tools.knowledge_tool import KnowledgeToolError


@pytest.fixture
def diagnoses():
    return [DiagnosisEvidence(cylinder_id=i, fault_type="suction_light", confidence=0.8,
        model_name="test_model", model_version="test_version", latency_ms=10., trace_id="tool-trace",
        warnings=["uncalibrated test score"]) for i in (1, 2, 3)]


@pytest.fixture
def evidence():
    path = Path(__file__).resolve().parents[1] / "knowledge/processed/chunks/knowledge_chunks.jsonl"
    chunk = KnowledgeChunk.model_validate_json(path.read_text(encoding="utf-8").splitlines()[0])
    return Evidence(chunk_id=chunk.chunk_id, text=chunk.text, score=0.8, metadata=chunk.metadata)


def context(**kwargs):
    return MaintenanceContext(file_path="test.xlsx", **kwargs)


def test_successful_diagnosis_then_knowledge_preserves_facts(diagnoses, evidence):
    order = []
    def diagnose(**kwargs):
        order.append("diagnosis")
        assert kwargs["spm"] == 90
        return diagnoses
    def retrieve(**kwargs):
        order.append("knowledge")
        assert kwargs["equipment_type"] == "drilling_pump"
        assert "suction_light" in kwargs["query"]
        return [evidence]
    before = deepcopy(diagnoses)
    report = MaintenanceAgent(diagnose, retrieve).run(context())
    assert order == ["diagnosis", "knowledge"]
    assert report.status == "completed"
    assert report.diagnosis == before == diagnoses
    assert report.citations == [evidence]
    assert report.diagnosis[0].trace_id == "tool-trace" != report.trace_id
    assert report.assessment.possible_causes == []
    assert report.assessment.risk_level == "UNKNOWN"
    assert report.maintenance.inspection_steps == []
    assert report.total_latency_ms >= 0
    assert MaintenanceReport.model_validate_json(report.model_dump_json()) == report


def test_normal_without_query_skips_knowledge(diagnoses):
    for d in diagnoses:
        d.fault_type = "normal_normal"
    knowledge = Mock()
    report = MaintenanceAgent(Mock(return_value=diagnoses), knowledge).run(context())
    assert report.status == "completed" and not report.knowledge_requested
    assert report.citations == [] and len(report.tool_calls) == 1
    knowledge.assert_not_called()


def test_normal_with_explicit_query_calls_knowledge(diagnoses, evidence):
    for d in diagnoses:
        d.fault_type = "normal_normal"
    knowledge = Mock(return_value=[evidence])
    report = MaintenanceAgent(Mock(return_value=diagnoses), knowledge).run(context(query="检修前的安全要求", top_k=3))
    assert report.knowledge_requested and report.status == "completed"
    knowledge.assert_called_once_with(query="检修前的安全要求", equipment_type="drilling_pump", top_k=3)


@pytest.mark.parametrize("error", [DiagnosisToolError("model_error", "private", "error-trace", 1.),
                                  RuntimeError("private endpoint")])
def test_diagnosis_failure_stops_workflow(error):
    knowledge = Mock()
    report = MaintenanceAgent(Mock(side_effect=error), knowledge).run(context())
    assert report.status == "diagnosis_failed"
    assert report.diagnosis == report.citations == []
    assert len(report.tool_calls) == 1 and report.tool_calls[0].status == "error"
    assert "private" not in report.model_dump_json()
    knowledge.assert_not_called()


def test_empty_knowledge_is_explicit_degradation(diagnoses):
    report = MaintenanceAgent(Mock(return_value=diagnoses), Mock(return_value=[])).run(context())
    assert report.status == "insufficient_evidence"
    assert report.diagnosis == diagnoses and not report.citations
    assert report.tool_calls[-1].status == "empty"


@pytest.mark.parametrize("error", [KnowledgeToolError("vector_store_error", "private path"), TimeoutError("private")])
def test_knowledge_exception_preserves_diagnosis(error, diagnoses):
    report = MaintenanceAgent(Mock(return_value=diagnoses), Mock(side_effect=error)).run(context())
    assert report.status == "knowledge_error"
    assert report.diagnosis == diagnoses and not report.citations
    assert report.tool_calls[-1].status == "error"
    assert "private" not in report.model_dump_json()


@pytest.mark.parametrize("bad_result", [[], {}, [None]])
def test_invalid_diagnosis_result_is_failure(bad_result):
    knowledge = Mock()
    report = MaintenanceAgent(Mock(return_value=bad_result), knowledge).run(context())
    assert report.status == "diagnosis_failed"
    knowledge.assert_not_called()


def test_wrong_equipment_evidence_is_not_cited(diagnoses, evidence):
    evidence.metadata.equipment_type = "compressor"
    report = MaintenanceAgent(Mock(return_value=diagnoses), Mock(return_value=[evidence])).run(context())
    assert report.status == "knowledge_error" and report.citations == []


def test_invalid_context_fails_before_tools():
    diagnose = Mock()
    with pytest.raises(ValidationError):
        MaintenanceAgent(diagnose, Mock()).run({"file_path": "test.xlsx", "spm": 110})
    diagnose.assert_not_called()

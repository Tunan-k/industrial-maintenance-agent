from copy import deepcopy
import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import UUID

import pytest
from fastapi.testclient import TestClient

from app.dependencies import CLASS_MAP_PATH
from app.services.diagnosis_service import DiagnosisService, DiagnosisServiceError
from app.services.model_service import ModelServiceError
from app.services.preprocessing_service import PreprocessingError
from app.tools.diagnosis_tool import DiagnosisTool, DiagnosisToolError, diagnose_equipment


@pytest.fixture
def raw():
    return dict(equipment_type="drilling_pump", spm=90, model_name="DSMT1DCNN10",
                device="cpu", file_name="test.csv", rows_used=2048, window_size=1024, stride=512,
                cylinders=[dict(cylinder_id=i, class_id=0, fault_type="normal", model_score=0.8,
                                window_agreement=0.7, window_count=3, dropped_nonfinite=1 if i == 1 else 0,
                                top_predictions=[dict(class_id=0, fault_type="normal", probability=0.8)])
                           for i in (1, 2, 3)])


@pytest.fixture
def setup(tmp_path, raw):
    input_file = tmp_path / "test.csv"
    input_file.write_text("fixture input")
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"unit test model identity")
    scaler = tmp_path / "scaler.joblib"
    scaler.write_bytes(b"unit test scaler")
    service = SimpleNamespace(model_service=SimpleNamespace(checkpoint_path=checkpoint),
                              scaler_path=scaler, diagnose_file=Mock(return_value=deepcopy(raw)))
    factory = Mock(return_value=service)
    return input_file, service, factory


def test_standard_function_maps_facts_and_provenance(setup, raw, monkeypatch):
    path, service, factory = setup
    monkeypatch.setattr("app.tools.diagnosis_tool.get_diagnosis_service", factory)
    evidence = diagnose_equipment(str(path), 90, "drilling_pump")
    assert [e.cylinder_id for e in evidence] == [1, 2, 3]
    assert all(e.confidence == 0.8 and e.fault_type == "normal" for e in evidence)
    assert evidence[0].model_version == "sha256:" + hashlib.sha256(b"unit test model identity").hexdigest()
    assert len(evidence[0].warnings) == 2 and len(evidence[1].warnings) == 1
    UUID(evidence[0].trace_id)
    assert len({e.trace_id for e in evidence}) == 1
    assert evidence[0].trace_id != diagnose_equipment(str(path))[0].trace_id
    assert service.diagnose_file.return_value == raw


@pytest.mark.parametrize("overrides", [
    {"file_path": "missing.xlsx"}, {"file_path": " "}, {"spm": 110},
    {"spm": True}, {"spm": "90"}, {"spm": 90.0}, {"equipment_type": "compressor"},
])
def test_invalid_input_does_not_load_model(setup, overrides):
    path, _, factory = setup
    with pytest.raises(DiagnosisToolError) as caught:
        DiagnosisTool(factory).diagnose(**({"file_path": str(path)} | overrides))
    assert caught.value.code == "invalid_input"
    UUID(caught.value.trace_id)
    assert caught.value.latency_ms >= 0
    factory.assert_not_called()


def test_real_missing_checkpoint_maps_to_model_error(setup, tmp_path):
    path, _, _ = setup
    factory = lambda: DiagnosisService(checkpoint_path=tmp_path / "missing.pt",
                                      class_map_path=CLASS_MAP_PATH, scaler_path=tmp_path / "scaler.joblib")
    with pytest.raises(DiagnosisToolError) as caught:
        DiagnosisTool(factory).diagnose(str(path))
    assert caught.value.code == "model_error"
    assert isinstance(caught.value.__cause__, ModelServiceError)
    assert str(tmp_path) not in str(caught.value)


def test_existing_unsupported_file_is_invalid_input(setup, tmp_path):
    _, _, factory = setup
    path = tmp_path / "unsupported.txt"
    path.write_text("not a supported diagnostic format")
    with pytest.raises(DiagnosisToolError) as caught:
        DiagnosisTool(factory).diagnose(str(path))
    assert caught.value.code == "invalid_input"
    factory.assert_not_called()


@pytest.mark.parametrize("cause,code", [
    (PreprocessingError("invalid columns"), "invalid_input"),
    (ModelServiceError("bad tensor"), "inference_error"),
    (TimeoutError("deadline expired"), "inference_error"),
])
def test_wrapped_service_failures_and_timeout(setup, cause, code):
    path, service, factory = setup
    wrapped = DiagnosisServiceError("service failed")
    wrapped.__cause__ = cause
    service.diagnose_file.side_effect = wrapped
    with pytest.raises(DiagnosisToolError) as caught:
        DiagnosisTool(factory).diagnose(str(path))
    assert caught.value.code == code
    assert caught.value.__cause__ is wrapped
    if isinstance(cause, TimeoutError):
        assert "timed out" in str(caught.value)


def test_model_initialization_timeout(setup):
    path, _, factory = setup
    factory.side_effect = TimeoutError("load deadline")
    with pytest.raises(DiagnosisToolError) as caught:
        DiagnosisTool(factory).diagnose(str(path))
    assert caught.value.code == "inference_error"


@pytest.mark.parametrize("invalid", ["nan", "duplicate_cylinder", "wrong_equipment"])
def test_invalid_service_results_never_become_evidence(setup, invalid):
    path, service, factory = setup
    result = service.diagnose_file.return_value
    if invalid == "nan":
        result["cylinders"][0]["model_score"] = float("nan")
    elif invalid == "duplicate_cylinder":
        result["cylinders"][1]["cylinder_id"] = 1
    else:
        result["equipment_type"] = "compressor"
    with pytest.raises(DiagnosisToolError) as caught:
        DiagnosisTool(factory).diagnose(str(path))
    assert caught.value.code == "inference_error"


def test_fastapi_upload_contract_is_unchanged(raw, monkeypatch):
    import app.main as main
    service = SimpleNamespace(diagnose_file=Mock(return_value=deepcopy(raw)))
    factory = Mock(return_value=service)
    monkeypatch.setattr(main, "get_diagnosis_service", factory)
    with TestClient(main.app) as client:
        response = client.post("/diagnose/file", files={"file": ("input.csv", b"fixture", "text/csv")},
                               data={"spm": "90"})
    assert response.status_code == 200, response.text
    result = response.json()
    assert set(result) == set(raw)
    assert result["file_name"] == "input.csv"
    assert result["cylinders"] == raw["cylinders"]
    assert "evidence" not in result
    factory.cache_clear.assert_called_once_with()

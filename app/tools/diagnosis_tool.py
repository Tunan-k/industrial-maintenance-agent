"""Diagnosis evidence adapter; all preprocessing and inference remain in the service."""
from __future__ import annotations

import hashlib
from pathlib import Path
from time import perf_counter
from typing import Callable, Literal
from uuid import uuid4

from langchain_core.tools import tool  # Existing legacy entry point only.
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.dependencies import get_diagnosis_service
from app.schemas.diagnosis import DiagnosisFileResponse
from app.services.diagnosis_service import DiagnosisService
from app.services.preprocessing_service import PreprocessingError


class DiagnosisToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)
    file_path: str = Field(min_length=1)
    spm: int = Field(default=90, ge=90, le=90)
    equipment_type: Literal["drilling_pump"] = "drilling_pump"
    start_time: str | None = None
    end_time: str | None = None


class DiagnosisEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    cylinder_id: int = Field(ge=1, le=3)
    fault_type: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1, allow_inf_nan=False)
    model_name: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    latency_ms: float = Field(ge=0, allow_inf_nan=False)
    trace_id: str = Field(min_length=1)
    warnings: list[str]


class DiagnosisToolError(RuntimeError):
    def __init__(self, code: str, message: str, trace_id: str, latency_ms: float):
        super().__init__(message)
        self.code = code
        self.trace_id = trace_id
        self.latency_ms = latency_ms


def _caused_by(exc: Exception, kind: type[Exception]) -> bool:
    seen = set()
    while exc is not None and id(exc) not in seen:
        if isinstance(exc, kind):
            return True
        seen.add(id(exc))
        exc = exc.__cause__
    return False


class DiagnosisTool:
    def __init__(self, service_factory: Callable[[], DiagnosisService] | None = None):
        self.service_factory = service_factory or get_diagnosis_service

    def _execute(self, file_path, spm=90, equipment_type="drilling_pump",
                 start_time=None, end_time=None):
        begin, trace_id = perf_counter(), str(uuid4())

        def failure(code, message):
            return DiagnosisToolError(code, message, trace_id, (perf_counter() - begin) * 1000)

        try:
            request = DiagnosisToolInput(file_path=file_path, spm=spm, equipment_type=equipment_type,
                                         start_time=start_time, end_time=end_time)
            path = Path(request.file_path)
            if not path.is_file() or path.suffix.lower() not in {".csv", ".xlsx"}:
                raise ValueError("Expected an existing CSV or XLSX file.")
        except (ValidationError, ValueError, OSError) as exc:
            raise failure("invalid_input", "Invalid diagnosis input file, equipment type or operating condition.") from exc

        try:
            service = self.service_factory()
            # Identify the checkpoint artifact, not a made-up release version.
            checkpoint = Path(service.model_service.checkpoint_path)
            with checkpoint.open("rb") as stream:
                version = "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()
            if not Path(service.scaler_path).is_file():
                raise FileNotFoundError("Missing scaler artifact.")
        except TimeoutError as exc:
            raise failure("inference_error", "Diagnosis timed out during initialization.") from exc
        except Exception as exc:
            raise failure("model_error", "Diagnosis model or preprocessing artifacts could not be loaded.") from exc

        try:
            raw = service.diagnose_file(file_path=path, spm=request.spm,
                                        start_time=request.start_time, end_time=request.end_time)
        except Exception as exc:
            if _caused_by(exc, TimeoutError):
                raise failure("inference_error", "Diagnosis inference timed out.") from exc
            if _caused_by(exc, PreprocessingError):
                raise failure("invalid_input", "Diagnosis input could not be preprocessed.") from exc
            # Loading is handled above. A model failure during execution is inference_error.
            raise failure("inference_error", "Diagnosis inference failed.") from exc

        try:
            validated = DiagnosisFileResponse.model_validate(raw)
            if validated.equipment_type != request.equipment_type or validated.spm != request.spm:
                raise ValueError("Mismatched diagnosis context.")
            ids = [c.cylinder_id for c in validated.cylinders]
            if sorted(ids) != [1, 2, 3]:
                raise ValueError("Expected one result for each of C1/C2/C3.")
            elapsed = (perf_counter() - begin) * 1000
            evidence = []
            for cylinder in validated.cylinders:
                warnings = ["confidence is the existing mean-window model_score; it is not a calibrated fault probability."]
                if cylinder.dropped_nonfinite:
                    warnings.append(f"Dropped {cylinder.dropped_nonfinite} non-finite windows before inference.")
                evidence.append(DiagnosisEvidence(
                    cylinder_id=cylinder.cylinder_id, fault_type=cylinder.fault_type,
                    confidence=cylinder.model_score, model_name=validated.model_name,
                    model_version=version, latency_ms=elapsed, trace_id=trace_id, warnings=warnings,
                ))
            return evidence, raw
        except Exception as exc:
            raise failure("inference_error", "Diagnosis service returned invalid evidence.") from exc

    def diagnose(self, file_path: str, spm: int = 90, equipment_type: str = "drilling_pump",
                 *, start_time: str | None = None, end_time: str | None = None) -> list[DiagnosisEvidence]:
        return self._execute(file_path, spm, equipment_type, start_time, end_time)[0]


def diagnose_equipment(file_path: str, spm: int = 90, equipment_type: str = "drilling_pump",
                       *, start_time: str | None = None, end_time: str | None = None) -> list[DiagnosisEvidence]:
    """Standard Python Tool interface, analogous to retrieve_knowledge."""
    return DiagnosisTool().diagnose(file_path, spm, equipment_type, start_time=start_time, end_time=end_time)


@tool("diagnose_industrial_equipment", args_schema=DiagnosisToolInput)
def diagnose_industrial_equipment(file_path: str, spm: int = 90,
                                 start_time: str | None = None, end_time: str | None = None,
                                 equipment_type: str = "drilling_pump") -> dict:
    """Legacy Tool adapter: preserve existing diagnosis fields and add standard evidence."""
    try:
        evidence, raw = DiagnosisTool()._execute(file_path, spm, equipment_type, start_time, end_time)
        return {**raw, "evidence": [e.model_dump(mode="json") for e in evidence]}
    except DiagnosisToolError as exc:
        return {"status": "error", "error_type": exc.code, "message": str(exc),
                "trace_id": exc.trace_id, "latency_ms": exc.latency_ms}

"""Contracts for the bounded, rule-based maintenance workflow."""
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.rag.schemas import Evidence
from app.tools.diagnosis_tool import DiagnosisEvidence, DiagnosisToolInput


class MaintenanceContext(DiagnosisToolInput):
    query: str | None = Field(default=None, min_length=1)
    top_k: int = Field(default=5, gt=0)


class ToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: Literal["diagnose_equipment", "retrieve_knowledge"]
    status: Literal["success", "empty", "error"]
    latency_ms: float = Field(ge=0, allow_inf_nan=False)
    error_code: str | None = None
    tool_trace_id: str | None = None


class MaintenanceAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    fault_judgment: str
    possible_causes: list[str] = Field(default_factory=list)
    risk_level: Literal["UNKNOWN"] = "UNKNOWN"
    risk_notes: list[str] = Field(default_factory=list)


class MaintenanceActions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    inspection_locations: list[str] = Field(default_factory=list)
    inspection_steps: list[str] = Field(default_factory=list)
    maintenance_suggestions: list[str] = Field(default_factory=list)


class MaintenanceReport(BaseModel):
    model_config = ConfigDict(extra="forbid")
    report_id: str
    trace_id: str
    created_at: datetime
    context: MaintenanceContext
    status: Literal["completed", "diagnosis_failed", "insufficient_evidence", "knowledge_error"]
    diagnosis: list[DiagnosisEvidence] = Field(default_factory=list)
    citations: list[Evidence] = Field(default_factory=list)
    knowledge_requested: bool = False
    knowledge_query: str | None = None
    knowledge_decision: str
    assessment: MaintenanceAssessment
    maintenance: MaintenanceActions = Field(default_factory=MaintenanceActions)
    warnings: list[str] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    total_latency_ms: float = Field(ge=0, allow_inf_nan=False)

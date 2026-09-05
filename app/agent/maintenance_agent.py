"""Two-tool deterministic workflow. No LLM, planner, or autonomous loop."""
from datetime import datetime, timezone
from time import perf_counter
from uuid import uuid4

from app.rag.schemas import Evidence
from app.schemas.maintenance import (
    MaintenanceActions, MaintenanceAssessment, MaintenanceContext, MaintenanceReport, ToolCall,
)
from app.tools.diagnosis_tool import DiagnosisEvidence, DiagnosisToolError, diagnose_equipment
from app.tools.knowledge_tool import KnowledgeToolError, retrieve_knowledge


class MaintenanceAgent:
    def __init__(self, diagnosis_tool=None, knowledge_tool=None):
        self.diagnosis_tool = diagnosis_tool if diagnosis_tool is not None else diagnose_equipment
        self.knowledge_tool = knowledge_tool if knowledge_tool is not None else retrieve_knowledge

    def run(self, context: MaintenanceContext) -> MaintenanceReport:
        # Revalidate mutable model instances as well as caller-provided dictionaries.
        context = MaintenanceContext.model_validate(
            context.model_dump() if isinstance(context, MaintenanceContext) else context)
        begin = perf_counter()
        trace_id, report_id = str(uuid4()), str(uuid4())
        calls, diagnoses, citations, warnings = [], [], [], []
        requested, query = False, None

        def finish(status, decision, judgment, suggestions):
            return MaintenanceReport(
                report_id=report_id, trace_id=trace_id, created_at=datetime.now(timezone.utc),
                context=context, status=status, diagnosis=diagnoses, citations=citations,
                knowledge_requested=requested, knowledge_query=query, knowledge_decision=decision,
                assessment=MaintenanceAssessment(fault_judgment=judgment,
                    risk_notes=["规则流程未建立风险分级或根因判定依据，风险等级为 UNKNOWN。"]),
                maintenance=MaintenanceActions(maintenance_suggestions=suggestions),
                warnings=list(dict.fromkeys(warnings)), tool_calls=calls,
                total_latency_ms=(perf_counter() - begin) * 1000,
            )

        stage = perf_counter()
        try:
            result = self.diagnosis_tool(file_path=context.file_path, spm=context.spm,
                equipment_type=context.equipment_type, start_time=context.start_time, end_time=context.end_time)
            if not isinstance(result, list):
                raise ValueError("Diagnosis Tool must return a list.")
            validated = [DiagnosisEvidence.model_validate(e.model_dump() if isinstance(e, DiagnosisEvidence) else e)
                         for e in result]
            if sorted(e.cylinder_id for e in validated) != [1, 2, 3]:
                raise ValueError("Missing or duplicate cylinder evidence.")
            diagnoses = validated
        except Exception as exc:
            calls.append(ToolCall(name="diagnose_equipment", status="error",
                error_code=exc.code if isinstance(exc, DiagnosisToolError) else "unexpected_tool_error",
                tool_trace_id=exc.trace_id if isinstance(exc, DiagnosisToolError) else None,
                latency_ms=(perf_counter() - stage) * 1000))
            warnings.append("诊断未成功完成，未调用知识检索，也未生成故障或维修结论。")
            return finish("diagnosis_failed", "diagnosis_failed", "无可用诊断结果。",
                          ["检查诊断输入或服务状态后重试。"])
        calls.append(ToolCall(name="diagnose_equipment", status="success",
            tool_trace_id=diagnoses[0].trace_id, latency_ms=(perf_counter() - stage) * 1000))
        warnings.extend(w for d in diagnoses for w in d.warnings)
        abnormal = [d for d in diagnoses if d.fault_type not in {"normal_normal", "normal"}]
        judgment = "；".join(f"C{d.cylinder_id}: {d.fault_type}" for d in diagnoses)
        if not abnormal and context.query is None:
            warnings.append("模型正常分类不等同于设备安全保证。")
            return finish("completed", "all_cylinders_normal_without_query", judgment,
                          ["保留本次诊断记录，按已有现场规程由工程师复核。"])

        requested = True
        decision = "explicit_query" if context.query is not None else "abnormal_diagnosis"
        faults = "、".join(dict.fromkeys(d.fault_type for d in abnormal))
        query = context.query or f"钻井泵 {faults} 故障原因 检查与维护参考"
        stage = perf_counter()
        try:
            result = self.knowledge_tool(query=query, equipment_type=context.equipment_type, top_k=context.top_k)
            if not isinstance(result, list) or len(result) > context.top_k:
                raise ValueError("Invalid Knowledge Tool result.")
            validated = [Evidence.model_validate(e.model_dump() if isinstance(e, Evidence) else e) for e in result]
            if len({e.chunk_id for e in validated}) != len(validated):
                raise ValueError("Duplicate knowledge evidence.")
            if any(e.metadata.equipment_type != context.equipment_type for e in validated):
                raise ValueError("Knowledge evidence equipment mismatch.")
            citations = validated
        except Exception as exc:
            calls.append(ToolCall(name="retrieve_knowledge", status="error",
                error_code=exc.code if isinstance(exc, KnowledgeToolError) else "unexpected_tool_error",
                latency_ms=(perf_counter() - stage) * 1000))
            warnings.append("知识检索失败；保留诊断事实，不生成无来源的维修步骤。")
            return finish("knowledge_error", decision, judgment,
                          ["由工程师复核诊断，并在知识服务恢复后重新检索。"])
        calls.append(ToolCall(name="retrieve_knowledge", status="success" if citations else "empty",
                              latency_ms=(perf_counter() - stage) * 1000))
        if not citations:
            warnings.append("未检索到证据，无法据此确认原因或给出具体维修措施。")
            return finish("insufficient_evidence", decision, judgment,
                          ["补充适用设备的经审核资料，并由工程师复核诊断。"])
        warnings.append("检索分数不是证据正确率；引用是待复核参考，规则流程不推断具体根因或维修步骤。")
        return finish("completed", decision, judgment,
                      ["逐条核对引用的正文、来源与适用范围，再由工程师制定检查和维护方案。"])


def run_maintenance(context: MaintenanceContext) -> MaintenanceReport:
    """Use the existing diagnosis service and explicitly configured Knowledge Tool."""
    return MaintenanceAgent().run(context)

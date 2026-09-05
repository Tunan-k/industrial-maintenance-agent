# V1 Phase 4-C：规则 Maintenance Workflow

## 实现范围

在已有空文件 app/agent/maintenance_agent.py 实现 MaintenanceAgent；新增 app/schemas/maintenance.py，提供 MaintenanceContext、MaintenanceReport 及其嵌套 schema。使用普通 Python 顺序调用现有 Diagnosis Tool 和 Knowledge Tool，每次最多两个 Tool 调用，无循环规划、LLM、LangGraph、Multi-Agent 或新框架。

```python
from app.agent.maintenance_agent import run_maintenance
from app.schemas.maintenance import MaintenanceContext

# Knowledge Tool 应在启动阶段按 Phase 4-A 方式绑定已有 Retriever。
report = run_maintenance(MaintenanceContext(
    file_path="demo_data/raw_data1.xlsx",
    equipment_type="drilling_pump", spm=90,
    start_time="15:35:00", end_time="16:02:00", top_k=3,
))
payload = report.model_dump(mode="json")
```

需要显式注入 Tool 时：`MaintenanceAgent(diagnosis_tool=..., knowledge_tool=...).run(context)`；参数接收普通 callable，例如 DiagnosisTool.diagnose、KnowledgeTool.retrieve_knowledge。

## Context 与规则

MaintenanceContext 复用 DiagnosisToolInput 的 file_path、spm、equipment_type 和可选 start_time/end_time；新增可选 query、top_k（默认 5）。非 90 SPM、非 drilling_pump、空查询或非法 top_k 在调用 Tool 前拒绝。

| 条件 | 行为 | Report status |
|---|---|---|
| 诊断失败或缺少有效 C1/C2/C3 | 不调用 Knowledge Tool，不生成诊断/维修结论 | diagnosis_failed |
| 全部 normal_normal/normal，且无用户 query | 跳过检索，保留诊断 | completed |
| 任一非正常分类，或用户提供 query | 带 equipment_type 调用 Knowledge Tool | 由检索结果决定 |
| 检索成功、有结果 | 保留全部诊断和原始引用 | completed |
| 检索成功、空结果 | 保留诊断，明确证据不足 | insufficient_evidence |
| 检索异常、无效响应或设备不匹配 | 保留诊断，禁止伪造引用 | knowledge_error |

默认检索词由钻井泵名称、原 fault_type 和“故障原因 检查与维护参考”组合；用户 query 存在时使用该明确查询。normal_normal 是当前真实 class map 的正常类别；normal 兼容普通 Tool 输出。未识别的故障标签不会被当作正常跳过检索。

## MaintenanceReport

- report_id、trace_id、created_at：本次流程标识和 UTC 时间。
- context：设备类型、SPM、输入与查询条件。
- diagnosis：原 DiagnosisEvidence 列表，保留三缸故障、confidence、模型版本及 Tool trace_id，不用流程 trace_id 覆写。
- citations：原 Evidence 列表，完整保留 chunk_id、text、score、metadata；source/page/section 可在 metadata 回查。
- knowledge_requested、knowledge_query、knowledge_decision：记录调用判断和实际查询。
- assessment：逐缸故障摘要、possible_causes、risk_level、risk_notes。
- maintenance：inspection_locations、inspection_steps、maintenance_suggestions。
- warnings、tool_calls、total_latency_ms：记录降级、原诊断警告、每个 Tool 的状态/错误码/耗时及总耗时。

**completed 表示规则流程完成，不代表已确认故障根因、证据相关性或可以执行维修。** 当前已有检索基线存在误召回，不能把相似度分数当成正确率。此阶段只做证据归集和流程建议，possible_causes、具体 inspection_steps 保持空，risk_level=UNKNOWN；需要人工核对适用范围后制定维修方案。正常分类也不构成设备安全保证。

这是一份规则生成的诊断与证据报告，不是 LLM 合成报告。本阶段没有新增 Web endpoint，因此也不把它描述为任务书全部 V1 Web 端到端 DoD 已完成。

## 测试与真实演示

新增 tests/test_maintenance_agent.py：13 项定向测试覆盖 Tool 顺序、原始事实不变、正常跳过、显式查询、诊断失败、空知识、Tool 异常、无效输出、设备不匹配、Context 校验及报告 JSON 往返。

新增 scripts/run_maintenance_demo.py，使用原 XLSX、原 DSMT 模型、原 multilingual-e5-small 与 Phase 3.7-B 持久化 Qdrant；不摄取或重新索引知识。沿用 Phase 4-A 的 EMBEDDING_MODEL_PATH、revision、query:/passage: 环境变量以及本地临时目录配置后运行：

```powershell
.\venv\Scripts\python.exe -B -m scripts.run_maintenance_demo --output reports/phase4_c_maintenance_report.json
$env:RUN_SEMANTIC_TESTS='1'
.\venv\Scripts\python.exe -B -m pytest -p no:cacheprovider -q
git diff --stat
git status --short
git diff --check
```

若 Knowledge Tool 未配置，新函数会报告 knowledge_error；显式演示脚本负责构造现有 Retriever 并注入 Tool。真实语料仍只有 46 个块已入索引，之前的 17 个超长 PDF 块未在本阶段处理。

## 变更文件

- app/schemas/maintenance.py：新增 Context、Report schema。
- app/agent/maintenance_agent.py：实现原有空占位文件。
- tests/test_maintenance_agent.py：新增 Workflow 测试。
- scripts/run_maintenance_demo.py：真实闭环演示入口。
- reports/phase4_c_maintenance_report.json：真实运行产物。
- PHASE4_C_WORKFLOW.md：本说明。

本阶段没有修改模型、RAG、现有两个 Tool 或 FastAPI。前序未提交变更与用户侧模型目录迁移保留。

## 真实闭环结果

真实示例报告：reports/phase4_c_maintenance_report.json。

- status=completed，调用顺序为 diagnose_equipment → retrieve_knowledge，二者均 success。
- C1=suction_light，confidence=0.7378732562065125。
- C2=suction_moderate，confidence=0.7900687456130981。
- C3=suction_severe，confidence=0.920985758304596。
- Workflow 耗时 82801.44 ms；不包含演示脚本预先加载 Embedding 的准备时间。
- 引用 3 条，均为钻井泵.pptx，page=null；包括已有提纲块。正文和 metadata 已逐项回查原始 chunk。
- 高分 PPT 召回问题仍在，completed 仅表示流程完成。报告保留原文、警告和 UNKNOWN 风险，未臆造原因或维修步骤。

本阶段已跟踪文件 diff 为 maintenance_agent.py +104/-0（原空占位文件）；另新增 schema 56 行、测试 123 行、演示脚本、运行报告和说明，未跟踪新增文件不计入默认 git diff --stat。整个工作区统计中的其他改动属于前序阶段或用户侧模型迁移。

## 全量测试与提交状态

pytest：**129 passed, 2 warnings in 124.16s**，RUN_SEMANTIC_TESTS=1，无跳过。两项警告仍为 TestClient/httpx 弃用提示及既有日期解析。

已执行 git diff --stat、git status --short、git diff --check。按此前要求尝试暂存当前及前序 Tool/RAG 工作，准备提交 feat: add embedding and qdrant vector store；git add 仍因 .git/index.lock: Permission denied 失败，未生成 commit。用户侧模型目录迁移未纳入暂存清单。

Phase 4-C 完成后停止，不接入 LLM，不扩展新框架。

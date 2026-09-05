# V1 Phase 4-B：Diagnosis Tool 标准化

## 接口与输出

普通 Python 接口与 Knowledge Tool 的使用方式对应，不需要创建 Agent：

```python
from app.tools.diagnosis_tool import diagnose_equipment, DiagnosisToolError

try:
    evidence = diagnose_equipment(
        file_path="demo_data/raw_data1.xlsx",
        spm=90,
        equipment_type="drilling_pump",
        start_time="15:35:00",  # 保留既有可选时间范围
        end_time="16:02:00",
    )
    payload = [item.model_dump(mode="json") for item in evidence]
except DiagnosisToolError as exc:
    error = {"error_type": exc.code, "message": str(exc),
             "trace_id": exc.trace_id, "latency_ms": exc.latency_ms}
```

也可使用 `DiagnosisTool(service_factory).diagnose(...)` 注入服务工厂进行组合或测试。默认复用已有 get_diagnosis_service 缓存，不重复实现预处理、模型加载或推理。

输出为 **list[DiagnosisEvidence]**，每个缸一项，严格保留 C1/C2/C3，不生成未经模型支持的整机故障。每项字段为：

| 字段 | 来源/含义 |
|---|---|
| cylinder_id | 现有缸编号 1/2/3 |
| fault_type | 现有模型聚合结果，原样保留 |
| confidence | 现有 model_score，不重算 softmax、不取窗口一致率代替 |
| model_name | DiagnosisService 的 model_name |
| model_version | 当前 checkpoint 文件的 sha256: 指纹，非臆造发布版本 |
| latency_ms | 本次输入验证、服务获取、版本标识与诊断的耗时；三缸共享一次调用耗时 |
| trace_id | 每次调用生成 UUID，同一调用三缸共享 |
| warnings | 明示 confidence 未校准；有丢弃非有限窗口时保留相应警告 |

checkpoint 指纹是模型权重工件标识，不代表训练代码、scaler、class map 的完整版本。confidence 是已有窗口平均类别得分，不可表述为真实故障概率。

## 异常契约

| code | 处理范围 |
|---|---|
| invalid_input | 不存在/不支持的文件，空路径，非 90 SPM，非 drilling_pump；输入预处理失败 |
| model_error | 服务初始化、checkpoint 读取、缺失 scaler 等模型工件加载失败 |
| inference_error | 推理失败、上游 TimeoutError、结果 schema 无效、缸重复/缺失或设备上下文不一致 |

异常携带 trace_id、latency_ms；安全外层消息不复制内部路径，`__cause__` 保留原始异常链。输入校验在模型获取前完成，非法输入不会触发模型加载。

超时测试使用直接及 DiagnosisServiceError 包装的 TimeoutError，验证其映射为 inference_error。本轮没有实现强制终止 PyTorch 推理的超时调度器，不能把模拟异常测试描述成可中断实际推理的硬截止时间。

## 兼容性

- 原 `diagnose_industrial_equipment.invoke(...)` 名称保留，使用项目已有 langchain_core 装饰器，没有新引入框架。成功保留原有响应字段并增加 evidence；业务异常返回 status=error 及标准错误码。旧装饰器入口的 schema 校验仍遵循其原有 ValidationError 行为；推荐新调用方使用普通 Python 接口获取统一 DiagnosisToolError。
- `/diagnose/file` 继续直接调用原 DiagnosisService，其输入、状态码与 DiagnosisFileResponse 未修改，HTTP 响应没有新增 evidence 字段。
- 未修改模型结构、训练、推理逻辑、Knowledge Tool、Chunk、Embedding、Qdrant，也没有进入 Agent 阶段。

## 文件与验证范围

- 修改 app/tools/diagnosis_tool.py：清理该文件中的旧注释版本，保留旧入口适配；新增标准输入、DiagnosisEvidence、错误映射和普通 Python 接口。
- 修改 tests/test_diagnosis_tool.py：保留真实 XLSX 的 C1/C2/C3 故障断言，增加 Evidence 与原始 fault_type/model_score 一致性及追溯字段断言。
- 新增 tests/test_diagnosis_evidence.py：非法文件/类型/SPM、真实缺失 checkpoint、预处理/推理错误、超时模拟、无效输出、原始事实映射和 FastAPI 响应兼容性。
- 新增本文；未修改其他项目实现文件。

真实 XLSX 样例仍使用 demo_data/raw_data1.xlsx 的 15:35:00–16:02:00，期望 C1=suction_light、C2=suction_moderate、C3=suction_severe。HTTP 合同测试使用替身服务验证请求与响应格式，真实模型验证由 XLSX Tool 测试完成，二者不混称真实 HTTP 端到端推理。

运行命令：在与 Phase 4-A 相同本地模型环境下，设置 RUN_SEMANTIC_TESTS=1、PYTHONDONTWRITEBYTECODE=1 和已有 F 盘临时目录，执行：

```powershell
.\venv\Scripts\python.exe -B -m pytest -p no:cacheprovider -q
git diff --stat
git status --short
git diff --check
```

首次定向测试发现 FastAPI 关闭阶段要求服务工厂提供 cache_clear；已补齐测试替身并保留关闭断言，未修改 FastAPI 生命周期。最终测试和 Git 状态见下方。

## 最终测试与 diff summary

全量命令：`.\venv\Scripts\python.exe -B -m pytest -p no:cacheprovider -q`，RUN_SEMANTIC_TESTS=1。

结果：**116 passed, 2 warnings in 144.46s**，无跳过。真实 XLSX 推理及逐缸 Evidence 一致性、非法文件、真实缺失模型、超时模拟、FastAPI 合同回归全部通过。警告为 TestClient/httpx 弃用提示和既有日期解析 dayfirst=False。

本阶段已跟踪文件 diff：

| 文件 | 新增行 | 删除行 |
|---|---:|---:|
| app/tools/diagnosis_tool.py | 136 | 219 |
| tests/test_diagnosis_tool.py | 14 | 1 |

合计 +150/-220；Diagnosis Tool 删除部分主要包括原文件中的历史注释实现，原推理服务未被重写。另新增未跟踪 tests/test_diagnosis_evidence.py（146 行）及本说明文件；git diff --stat 默认不计入未跟踪新增文件。

已执行 git diff --stat、git status --short、git diff --check，空白检查通过。整个工作区统计还包含前序未提交 RAG 修改及用户侧模型目录迁移，不能当作本阶段修改范围。

按此前要求尝试暂存当前及前序 Tool/RAG 变更并准备提交 feat: add embedding and qdrant vector store；git add 仍被 .git/index.lock: Permission denied 阻止，**没有生成提交**。用户侧诊断模型目录迁移不在暂存清单中。

Phase 4-B 完成后停止，不进入 Agent。

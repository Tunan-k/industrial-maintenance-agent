# Phase 3.6：Retriever 质量评估

已建立独立评估脚本、12 条固定工业查询和可回查的初始相关性标签，使用 Phase 3.5 的原有本地 multilingual-e5-small、384 维归一化向量、Qdrant 本地模式和原 Retriever。没有更换模型、修改 Chunk 或添加检索算法。

## 标签与指标口径

`tests/fixtures/retrieval_evaluation.json` 保存 query、expected_equipment_type、expected_source_type，以及每条查询对全部 63 个 chunk 的 0/1/2 标签和理由，共 756 项初始判断。由 Codex 对照已有文本建立，**pending_human_review，人工审阅者及时间为空**，不能当作已由工业专家验收的金标准。人工复核应同时检查正例、负例及“无直接答案”的判断，然后填写审阅身份和时间。冻结语料 SHA256 防止内容改变后继续套用旧标签。

- 2：正文直接支持所问答案；1：背景或部分支持；0：不支持。主指标只把 2 算相关。
- Top1 命中率 = 第 1 名标签为 2 的查询数 / 查询数。
- Top3 相关率采用 Precision@3 = 前 3 名中标签为 2 的总数 / (查询数 × 3)。不足 3 条的空位计未命中。
- 同时报告 Hit@3（至少一个直接相关结果的查询比例），避免把它误称为 Precision@3。
- 相关性独立于 metadata 匹配，不能因 equipment_type 相同或 score 高就判相关。
- expected_source_type 是语义期望：安全操作为 safety_procedure、论文为 research_paper、排障为 troubleshooting、处置为 maintenance_sop；内部项目说明沿用 other。现有源数据均为 other，此处没有修改它来迎合测试。

10 条中文问题含原来的 3 条运维问题，并覆盖安全隔离、泄压阀、采集、训练数据、信号处理及论文实验结果；另有 2 条同标签英文对照。查询和初始标签在本轮真实检索前保存。该小集合是开发基线，非独立盲测；英文问题是配对对照，不能把语言分组总分直接当成严格语言性能对比。

## 本轮真实结果

| 范围 | Top1 命中率 | Top3 相关率 | Hit@3 |
|---|---:|---:|---:|
| 全部 12 条 | 3/12 = 25.00% | 5/36 = 13.89% | 5/12 = 41.67% |
| 语料有直接答案的 9 条 | 3/9 = 33.33% | 5/27 = 18.52% | 5/9 = 55.56% |
| 中文 10 条 | 2/10 = 20.00% | 3/30 = 10.00% | 3/10 = 30.00% |
| 英文对照 2 条 | 1/2 = 50.00% | 2/6 = 33.33% | 2/2 = 100.00% |

完整逐 query Top-K 的 chunk_id、score、source、page、正文、metadata、标签和理由见 `reports/phase3_6_retrieval.json`；表格版见同名 `.md`。page=null 代表原文 metadata 无页码，未补造页码。分数是余弦相似度，不是诊断置信度或证据正确率。

## 误召回分析

### PPT 偏向：观察成立，成因尚未隔离

PPTX 只有 3/46 个已索引块（6.52%），却占前 3 名结果的 28/36（77.78%），其中 23 项标签为 0。中文上锁挂牌 q04 返回 3 个 PPT 块，没有安全内容；英文对照 q11 第 1 名命中 IADC 安全正文。中文 VMD 问题 q08 未召回直接步骤块，英文对照 q12 第 3 名召回 PDF 第 5 页步骤表。

这支持“当前排序对中文 PPT 内容存在偏向”的判断；语言、文档标题、块长度、领域词重复和语料数量差异可能共同影响排序。没有做独立消融，因此不能断言 PPT 文件格式或某一因素单独导致问题，也没有为改善分数临时改查询或模型。

### Chunk 问题：有直接证据

63 个原始块中仅 46 个满足现有模型长度限制，17 个 PDF 块被明确排除。PDF 入索引 36/53，HTML 7/7，PPTX 3/3。VMD 步骤正例 2 个仅入索引 1 个；角域重采样正例 5 个入索引 3 个；论文准确率正例 3 个入索引 1 个。这是召回覆盖的限制，不能全部归咎于排序。

`chk_c127fd4722c47b8806f6` 正文只有“汇 报 提 纲”，却在 8 条查询的 Top3 出现。英文安全查询同时召回版权/Cookie 页脚与电话联系方式；可见 HTML boilerplate 也影响结果，不只是 PPT。部分 PDF 描述跨页，如传感器信息在第 11/12 页分开，单块可能只支持部分答案。

另外，q01–q03 在初始审阅中没有满足完整故障原因/现场检查/处置问题的直接答案。项目介绍、实验造故障和信号处理方法不能冒充维修规程；这是知识覆盖缺口，单纯提高检索排序无法补足。

### Metadata 不足：已确认，不能单独解释排序

全部 63 个 source_type=other，丢失论文/安全指导/项目说明的类型区分。36 个返回结果 equipment_type 均匹配；但全部语料本来就只有 drilling_pump，这不能证明多设备混合语料效果。额外 compressor 过滤返回空列表，验证排除行为。

source_type 与查询语义期望仅匹配 6/36 项（两个期望 other 的问题），不代表这 6 项语义都相关。现 Retriever 只支持 equipment_type 过滤，本轮未添加来源类型过滤。31/36 个返回结果 page 缺失；63 个块 equipment_model 均为空，不能声称检索已实现 HH2400 型号级约束。原始 document_type/source_name 等仍可提供人工追溯线索。

metadata 不足阻碍类型/型号约束和定位，但当前并未把 source_type 用于排序，因而不能称其为已证实的排序误召回直接原因。

## 可复现运行

在仓库根目录 PowerShell 执行，沿用已经下载并验证的同一模型：

```powershell
$env:EMBEDDING_MODEL_PATH = 'F:\git-demo-file\industrial-maintenance-agent\artifacts\embedding\multilingual-e5-small'
$env:EMBEDDING_REVISION = '614241f622f53c4eeff9890bdc4f31cfecc418b3'
$env:EMBEDDING_QUERY_PREFIX = 'query: '
$env:EMBEDDING_DOCUMENT_PREFIX = 'passage: '
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:TEMP = 'F:\git-demo-file\industrial-maintenance-agent\artifacts\embedding\tmp'
$env:TMP = $env:TEMP
.\venv\Scripts\python.exe -B -m scripts.evaluate_retrieval --top-k 3
$env:RUN_SEMANTIC_TESTS = '1'
.\venv\Scripts\python.exe -B -m pytest -p no:cacheprovider -q
git diff --stat
git status
```

临时 Qdrant 索引随运行创建和关闭，不改动既有持久化业务集合。缺模型报错，禁止替代 embedding；只对现有明确的模型长度异常记录排除，其他异常传播。测试覆盖固定分母、部分相关不算命中、语料漂移、缺失/非法标签、伪造/重复 Evidence、翻译对照标签一致性，以及真实模型的过滤与来源回查。测试验证评估器可信运行，不以断言虚高质量来通过。

## 本轮文件

- 新增 `scripts/evaluate_retrieval.py`。
- 新增 `tests/fixtures/retrieval_evaluation.json`。
- 新增 `tests/test_retrieval_evaluation.py`。
- 修改 `scripts/validate_semantic_retrieval.py`，仅新增可传入查询列表的参数，默认 3 条查询保持不变。
- 新增 `reports/phase3_6_retrieval.json`、`reports/phase3_6_retrieval.md` 和本文。

本轮仅建立和运行评估。没有继续修复 Chunk/metadata，没有添加 Reranker、Hybrid Search 或 Agent。

## 验证与 Git 状态

2026-09-05 真实环境全量运行：**75 passed, 1 warning in 144.42s**，RUN_SEMANTIC_TESTS=1，未跳过真实模型测试。警告来自既有诊断预处理日期解析 dayfirst=False，本轮未修改。

已执行 `git diff --stat`、`git status`、`git diff --check`；空白检查通过。仓库存在用户侧的诊断模型目录迁移，未纳入本轮暂存清单。diff --stat 默认不统计未跟踪新增文件，本轮新增文件以本文清单及 git status 为准。

按要求尝试暂存本轮与尚未提交的 Phase 3.5 RAG 文件，以执行 `git commit -m "feat: add embedding and qdrant vector store"`。写权限申请已获授予，但 git add 仍报 `.git/index.lock: Permission denied`，因此暂存失败，**提交未生成**。未绕过文件保护，未改动用户的诊断模型迁移。

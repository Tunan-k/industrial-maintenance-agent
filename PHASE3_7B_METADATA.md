# Phase 3.7-B：Pilot metadata 修复与复测

## 实施结果

对现有 63 个 KnowledgeChunk 仅修正 metadata.source_type。文本、chunk_id、document_id、顺序及其余 metadata 均逐项验证不变；没有过滤任何目录或版权块。本轮没有修改 Embedding 模型或 app/rag/embedding.py，也没有修改 app/rag/retriever.py。

| source_type | 修改前 | 修改后 |
|---|---:|---:|
| other | 63 | 0 |
| research_paper | 0 | 53 |
| industry_safety_guidance | 0 | 7 |
| internal_project_note | 0 | 3 |
| ppt | 不支持 | 支持，当前 0 |

SourceType 兼容新增 industry_safety_guidance、internal_project_note、ppt，保留原有枚举。ppt 用于未明确业务类型的演示资料；现有 3 个 PPTX 块已明确为 internal_project_note，优先采用这一精确类型，source_format=pptx 保持不变。不虚构第四类来源，也不把已知内部项目说明改成笼统的 ppt。

同步修正 3 份持久化 KnowledgeDocument 的顶层 source_type，以保持后续 chunking 的可复现性；仅改该字段，不重新设计或重新解析文档。batch_ingestion 补充同一映射，避免下次摄取恢复 other。authority_level 仍为 3。

## 向量与 Qdrant

真实重新编码使用原有本地 intfloat/multilingual-e5-small，revision `614241f622f53c4eeff9890bdc4f31cfecc418b3`，384 维，CPU，归一化，query:/passage: 前缀。未使用替代 embedding。

持久化路径：`knowledge/vector_store/phase3_7b`；collection：`semantic_validation`。这是本轮独立的持久化 Pilot 索引，不覆盖未知业务集合。

**63 个原始块全部完成 metadata 修复，但只有 46 个符合现有模型长度限制并重新编码、写入 Qdrant。** 其余 17 个 PDF 块仍超过上限，与修改前排除清单完全相同；没有静默截断、填充伪向量或声称已写入全部 63 个块。逐块排除原因见评估 JSON。

写入结束关闭客户端后，独立重新打开 Qdrant，检查 46 个 point 的原始 ID、正文、全部 metadata、384 维有限向量，全部匹配修复后的原始 chunk。持久化 payload 类型为 research_paper 36、industry_safety_guidance 7、internal_project_note 3。数据库文件在既有 gitignore 范围内，不提交二进制索引。

## 原有 12 条查询对比

修改前基线为 `reports/phase3_6_retrieval.json`，修改后为 `reports/phase3_7b_retrieval.json`。保持原来的 12 条 query、expected_equipment_type、expected_source_type、756 项初始相关性标签及理由完全不变。只在验证 source_type 为唯一差异后更新评估语料指纹及版本，没有放宽指纹校验。

| 指标 | 修改前 | 修改后 | 差值 |
|---|---:|---:|---:|
| Top1 命中率 | 3/12 = 25.00% | 3/12 = 25.00% | 0 个百分点 |
| Top3 相关率（Precision@3） | 5/36 = 13.89% | 5/36 = 13.89% | 0 个百分点 |
| PPT 召回比例 | 28/36 = 77.78% | 28/36 = 77.78% | 0 个百分点 |

Top1/Top3 以初始标签 grade=2 为相关；PPT 比例按 source_format=pptx 统计 Top3 返回位置，不能改用 source_type=ppt（当前为 0）而制造召回比例下降的假象。

12 条查询的 Top3 chunk_id 及顺序全部相同，最大 score 差为 0。原因是当前 embedding 文本由 title、section、text 组成，不包含 source_type；Retriever 只用 equipment_type 过滤，不按 source_type/authority_level 排序。此次修复让来源 metadata 正确落库，**没有带来语义检索质量提升**。

原有 source_type 期望仍含 safety_procedure/other，与新增分类并非全都一一对应；因此本轮不把 source_type_match 当作检索质量改善指标。后续如需重标该字段，应另存版本，不改变当前比较基线。所有相关性标签仍为 Codex 初审、pending_human_review，不能宣称人工专家验收。

## 变更文件

- app/rag/schemas.py：兼容新增来源枚举。
- app/rag/source_metadata.py：最小来源类型映射。
- app/rag/batch_ingestion.py：摄取时明确传 source_type。
- scripts/repair_pilot_metadata.py：幂等 metadata 迁移，校验字段不变并更新评估指纹。
- scripts/validate_semantic_retrieval.py：增加可选持久化索引路径，原临时模式不变。
- scripts/evaluate_retrieval.py：透传索引路径、记录路径，报告标题通用化。
- tests/test_source_metadata.py：映射、幂等性、文本/ID/embedding 输入不变测试。
- knowledge/processed/chunks/knowledge_chunks.jsonl、3 份 documents/*.knowledge.json：仅 source_type 数据变化。
- tests/fixtures/retrieval_evaluation.json：语料指纹、版本、迁移说明变化；所有 cases 不变。
- reports/phase3_7b_retrieval.json、同名 .md、phase3_7b_comparison.json 及本文。

## 可复现命令

在仓库根目录 PowerShell 执行，沿用既有模型路径：

```powershell
$env:EMBEDDING_MODEL_PATH='F:\git-demo-file\industrial-maintenance-agent\artifacts\embedding\multilingual-e5-small'
$env:EMBEDDING_REVISION='614241f622f53c4eeff9890bdc4f31cfecc418b3'
$env:EMBEDDING_QUERY_PREFIX='query: '
$env:EMBEDDING_DOCUMENT_PREFIX='passage: '
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONIOENCODING='utf-8'
$env:TEMP='F:\git-demo-file\industrial-maintenance-agent\artifacts\embedding\tmp'
$env:TMP=$env:TEMP
.\venv\Scripts\python.exe -B -m scripts.repair_pilot_metadata
.\venv\Scripts\python.exe -B -m scripts.evaluate_retrieval --qdrant-path knowledge/vector_store/phase3_7b --output reports/phase3_7b_retrieval.json
$env:RUN_SEMANTIC_TESTS='1'
.\venv\Scripts\python.exe -B -m pytest -p no:cacheprovider -q
git diff --stat
git status
```

本轮停止于 metadata 修复与复测，没有引入 Reranker、Hybrid Search、GraphRAG，也没有继续处理超长块或噪声过滤。

## 最终验证与 Git

全量测试：**83 passed, 1 warning in 114.03s**，RUN_SEMANTIC_TESTS=1，无跳过。警告来自既有诊断日期解析 dayfirst=False。metadata 定向测试、原 63 块重生成一致性测试、真实语义评估均通过。

已执行 git diff --stat、git status、git diff --check。diff --stat 含本轮前已经存在的 Phase 3.5/3.6 修改和用户侧诊断模型迁移，未跟踪新增文件不计入该统计；本轮变更以上文清单为准。

已按要求尝试暂存本轮及尚未提交的前序 RAG 文件，准备提交信息 feat: add embedding and qdrant vector store；git add 仍因 .git/index.lock: Permission denied 失败，因此未生成 commit。用户侧诊断模型迁移不在暂存清单中。

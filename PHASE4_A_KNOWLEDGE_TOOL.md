# V1 Phase 4-A：Knowledge Tool

新增 `app/tools/knowledge_tool.py`，复用已有 Retriever 和 Evidence，不引入 LLM、Agent/LangGraph、Reranker 或新的依赖。Embedding、Chunk、Qdrant、Retriever 核心代码未改动。

## 接口

```python
from app.tools.knowledge_tool import configure_knowledge_tool, retrieve_knowledge

# 启动阶段绑定已经构造好的 Retriever；模型、向量库生命周期由调用方管理。
configure_knowledge_tool(existing_retriever)
evidence = retrieve_knowledge(
    query="钻井泵阀盒部署哪些传感器，采样频率是多少？",
    equipment_type="drilling_pump",
    top_k=3,
)
payload = [item.model_dump(mode="json") for item in evidence]
```

正式签名：`retrieve_knowledge(query, equipment_type=None, top_k=5) -> list[Evidence]`。

每个 Evidence 保留 chunk_id、text、score、metadata。equipment_type 存在时传给 Retriever 的 filters；省略时不推断设备、不附加过滤。严格输入 schema 拒绝空白查询、空白设备、非整数/非正 top_k（包括 bool）。正常无命中返回空列表。

需要多个独立 Retriever 时使用 `KnowledgeTool(retriever).retrieve_knowledge(...)`，避免修改默认绑定。默认函数不隐式加载模型、不创建或重建索引。`configure_knowledge_tool(None)` 解除绑定，不关闭调用方资源。

## 异常契约

抛出 KnowledgeToolError，`.code` 可稳定判断错误；错误与正常空结果保持不同语义。

| code | 场景 |
|---|---|
| invalid_input | 输入不满足 KnowledgeToolInput |
| not_configured | 尚未绑定 Retriever |
| embedding_error | 现有 EmbeddingError |
| vector_store_error | 现有 VectorStoreError |
| retrieval_error | 其他检索异常 |

外层消息不复制内部模型路径/服务地址，原始异常通过 `__cause__` 保留供调试。调用者可以捕获错误码决定展示或降级，不返回伪造 Evidence。

## 测试与真实调用

新增 `tests/test_knowledge_tool.py`，覆盖默认参数、过滤转发、Evidence JSON 合同、空结果、非法输入、未配置、后端异常与原始异常链。真实测试由 RUN_SEMANTIC_TESTS=1 启用，使用原 multilingual-e5-small 与 Phase 3.7-B 已持久化 Qdrant，通过公开函数运行中文查询，并核对结果正文和 metadata 与真实 chunk 一致；compressor 过滤验证空结果。

真实 Tool 测试只编码查询，不生成新的 chunk 向量、不重建索引。现有完整测试套件中的旧语义评估仍会建立自己的临时索引。当前持久化 Pilot 保留 46 个可编码块；原来的 17 个超长块不因新增 Tool 自动变得可检索，Tool 也不表示检索质量已经提升。

在 Phase 3.7-B 相同模型环境配置下运行：

```powershell
$env:RUN_SEMANTIC_TESTS='1'
# 可选：KNOWLEDGE_QDRANT_PATH 指向已有 Phase 3.7-B 索引目录。
.\venv\Scripts\python.exe -B -m pytest -p no:cacheprovider -q tests/test_knowledge_tool.py
.\venv\Scripts\python.exe -B -m pytest -p no:cacheprovider -q
git diff --stat
git status
```

本轮变更仅新增 Knowledge Tool、对应测试和本文；前序未提交的 RAG 文件及用户侧诊断模型迁移不是本轮实现内容。完成后停止，不进入 LLM 或 Agent 阶段。

## 本轮验证结果

完整 pytest：**98 passed, 1 warning in 136.43s**。RUN_SEMANTIC_TESTS=1，真实 Knowledge Tool 调用与全部既有真实语义测试均运行，无跳过。唯一警告仍为既有诊断预处理日期解析。

已执行 git diff --stat、git status --short、git diff --check。按要求尝试暂存本轮及前序未提交 RAG 文件，以提交 feat: add embedding and qdrant vector store；仍被 .git/index.lock: Permission denied 阻止，未生成提交。未暂存用户侧诊断模型目录迁移。

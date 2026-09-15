# 初步 Code Review Report

- 审查日期：2026-04-12
- 审查类型：核心链路初步审查（只读，未修改正式代码）
- 验证：`D:\conda_envs\rag-customer-service\python.exe -m pytest -q`
- 结果：140 passed，1 条第三方弃用警告

> **审查范围说明**：本报告覆盖入口 API、ChatService、RAG、检索、MCP、商品上下文、会话、摄取/Qdrant、配置及核心测试。它不是逐文件完成的全仓库最终审计；未覆盖的部分不代表没有问题。

## 真实调用链

`POST /chat` → `app.api.chat.chat` → 全局 `ChatService` → 商品上下文解析 / `route_query`：

- 普通问题：`RAGService.chat` → Dense + BM25 → RRF → Reranker → AnswerReliability → ContextBuilder → DeepSeek
- 订单问题：`MCPService` → stdio MCP `get_order` → DeepSeek
- 订单 + 规则：并发 `RAGService.prepare` 与 MCP → DeepSeek combined
- 闲鱼商品问题：MCP `get_item_info` → 指定商品范围 RAG → DeepSeek

---

## P0：必须立即修复

当前未发现已确认的 P0。

## P1：核心功能、可用性或回答正确性问题

### P1-01：普通 RAG 链路在 FastAPI 事件循环中执行阻塞操作

- **文件 + 行号**：`app/services/chat_service.py:365`；`app/services/rag_service.py:115-131`
- **当前代码行为**：异步 `chat_async()` 直接调用同步 `self.chat(query)`；同步链路内会运行 embedding、Qdrant、reranker 和 OpenAI 调用。
- **触发条件**：任意普通 RAG 请求；模型加载、检索或 LLM 响应变慢时更明显。
- **实际影响**：事件循环被阻塞，单 worker 中其他 `/chat`、`/health` 请求会排队，严重时超时；并发能力显著下降。
- **证据**：`chat_async` 为 async 方法，但第 365 行没有使用 `asyncio.to_thread`；`RAGService.chat` 是同步方法并执行完整链路。
- **建议修改方向**：将普通 RAG 完整链路放入线程池，或改造为真正的异步客户端/模型执行路径；避免在 async API handler 中直接运行同步 I/O 与 CPU 推理。
- **置信度**：高

### P1-02：普通 RAG 依赖失败会直接传播为 HTTP 500，缺少统一降级

- **文件 + 行号**：`app/api/chat.py:95-102`；`app/services/chat_service.py:365`；`app/services/rag_service.py:117-131`
- **当前代码行为**：API 直接等待 `chat_async()`；普通 RAG 的检索、模型和生成过程没有业务级异常兜底。
- **触发条件**：Qdrant 不可用、Embedding/Reranker 模型加载失败、Qdrant 请求失败、DeepSeek 超时或 API 失败。
- **实际影响**：用户收到未结构化 HTTP 500，而不是稳定的“稍后重试/转人工”响应；与订单和闲鱼分支已有降级行为不一致。
- **证据**：`RAGService.chat()` 内 `prepare()` 与 `generate()` 没有异常转换；API 路由也未处理这类预期依赖故障。
- **建议修改方向**：在 RAG service 边界捕获并区分预期依赖异常，返回统一的 `can_answer=False`、`next_step=human_handoff` 结构；保留完整错误日志。
- **置信度**：高

### P1-03：Reliability 只检查第一条证据，但 ContextBuilder 会拼入全部 rerank 结果

- **文件 + 行号**：`app/rag/pipeline.py:44-55`；`app/rag/answerability.py:55-82`；`app/rag/context_builder.py:33-56`
- **当前代码行为**：只要 Top-1 rerank score 达标，就将全部 reranked chunks 传入 ContextBuilder；低分、无关、重复或冲突 chunk 不会被排除。
- **触发条件**：第一条召回相关，但其余 Top-K 属于其他主题、低相关或含冲突规则时。
- **实际影响**：正确证据已检索到时，低质量上下文仍会污染 LLM 输入，造成混合、错误或无法可靠归因的回答。
- **证据**：`AnswerReliability.evaluate()` 以第一条有效分数返回结论；`run_after_rerank()` 随后调用 `context_builder.build(rerank_results)`，未做分数过滤、去重或限制。
- **建议修改方向**：构建上下文前按分数阈值、Top-1 分差、最大 chunk 数、内容去重和主题/文档范围筛选；sources 应只返回实际进入上下文的证据。
- **置信度**：高

---

## P2：稳定性、数据边界或维护性问题

### P2-01：空 query 被 API 接受，后续可能产生无效检索或 500

- **文件 + 行号**：`app/api/chat.py:19`；`app/generation/prompt.py:32-35`
- **当前代码行为**：`query` 仅声明为 `str`，允许空字符串和纯空白字符串。
- **触发条件**：例如 `{ "query": "", "chat_id": "review" }`。
- **实际影响**：请求进入检索流程；若走到生成，prompt 层会抛出 `ValueError("query must not be empty")`，可能转为 HTTP 500。
- **证据**：实际构造 `ChatRequest(query='', chat_id='review')` 成功；`build_messages()` 明确拒绝空 query。
- **建议修改方向**：为 `query` 设置最小长度，并增加去除空白后的 validator；可返回 422 或稳定的澄清响应。
- **置信度**：高

### P2-02：会话状态只保存在进程内，且没有 TTL、总量限制或并发控制

- **文件 + 行号**：`app/api/chat.py:11`；`app/services/session_manager.py:17-35`、`53-70`
- **当前代码行为**：模块级 `ChatService` 持有 `SessionManager`，其 sessions 是永久增长的内存字典；仅限制单会话 history 条数。
- **触发条件**：大量不同 `chat_id`、服务重启、多 worker 部署、同一 chat_id 并发请求。
- **实际影响**：内存持续增长；重启丢失订单/商品上下文；多 worker 下同一会话状态不一致；并发请求可能使轮次顺序和 state 更新不确定。
- **证据**：`self.sessions` 无删除、TTL、容量上限、锁或外部存储实现。
- **建议修改方向**：明确会话生命周期；生产部署使用 Redis/数据库等共享存储，增加 TTL、容量限制，以及同会话顺序控制。
- **置信度**：高

### P2-03：Qdrant 清理逻辑可能删除共享 collection 中的非本知识库 points

- **文件 + 行号**：`app/infrastructure/qdrant.py:124-147`
- **当前代码行为**：只要 point 的 `source` 是相对路径且后缀为 `.md` 或 `.txt`，就会被识别为当前知识库遗留数据并删除。
- **触发条件**：同一个 collection 被其他语料域、实验数据或服务复用。
- **实际影响**：一次 ingestion 可能误删不属于当前知识库的向量数据。
- **证据**：第 130-132 行没有验证 corpus、tenant、ingestion namespace 或所属知识库标识。
- **建议修改方向**：在 payload 增加 corpus/tenant/ingestion namespace，并只删除匹配该标识的数据；或将 collection 明确设计为单一语料专用。
- **置信度**：高

### P2-04：Reranker threshold 未见与实际模型绑定的校准依据

- **文件 + 行号**：`config/settings.py:29`；`app/rag/answerability.py:31-48`
- **当前代码行为**：所有配置的 CrossEncoder 都使用一个固定阈值 `0.5`。
- **触发条件**：更换 reranker 模型、模型输出为 logit 而非概率、知识库或查询分布变化。
- **实际影响**：可能出现已检索正确 chunk 却被拒答，或低质量证据被放行。
- **证据**：模型路径可由 `RERANKER_MODEL_PATH` 替换，但没有发现模型版本绑定、分数分布记录、阈值校准报告或可靠性评测依据。
- **建议修改方向**：使用固定验证集，按模型版本记录阈值、precision/recall、拒答率和错误类型；不要假设不同模型的 raw score 可直接比较。
- **置信度**：中
- **需要进一步验证**：需在实际部署的 reranker、真实知识库和代表性问题集上完成分数校准。

---

## 当前最严重的 5 个问题

1. 普通 RAG 在 async 事件循环内阻塞执行。
2. 普通 RAG 依赖异常会裸露为 HTTP 500。
3. 高分 Top-1 放行后，低分/无关 chunk 仍可进入 LLM context。
4. 内存会话没有 TTL、共享存储或并发控制。
5. Qdrant 清理范围过宽，存在共享 collection 误删风险。

## 最容易出错的调用链

`POST /chat` → 普通 RAG → 同步 embedding / Qdrant / reranker / DeepSeek。

该链路同时包含事件循环阻塞、依赖异常未降级、空 query 未拒绝，以及低质量证据可混入 context 的风险。

## 已实现较正确、不建议仅因本次 Review 而修改的部分

- MCP client 使用 context manager，连接关闭路径明确。
- MCP 工具返回结果做了结构字段校验。
- 商品编号冲突时会明确澄清，不会静默切换商品。
- 闲鱼 RAG 同时使用 Qdrant 和 BM25 的 scope 过滤，避免其他商品内容串入。
- ingestion manifest 在 Qdrant 成功后才原子写入。
- 未支持的订单操作存在明确拒绝路径。

## 建议修复顺序

1. 修复 P1-01：让普通 RAG 非阻塞执行。
2. 修复 P1-02：统一普通 RAG 的异常降级和 API 响应。
3. 修复 P1-03：限制进入 ContextBuilder 的证据并去重。
4. 修复 P2-01：拒绝空白 query，并增加回归测试。
5. 根据部署方式设计 P2-02 的会话存储与生命周期。
6. 确认 collection 使用边界后修复 P2-03。
7. 通过真实数据完成 P2-04 阈值校准。

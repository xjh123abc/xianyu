# 深度 Code Review 报告

**范围**：当前工作区代码；未修改任何正式代码。  
**验证**：`python3 -m compileall -q app config mcp_servers scripts eval` 通过。当前环境缺少 `pytest`，无法执行测试集。

## 真实调用链

```text
POST /chat
→ app/api/chat.py
→ ChatService.chat_async()
→ SQLite SessionManager 会话锁/历史
→ 订单路由 / 商品上下文与意图判断
  ├─ 普通 RAG：RAGService
  │   → VectorSearch + BM25Search
  │   → HybridSearch(RRF)
  │   → Reranker
  │   → AnswerReliability
  │   → RAGPipeline / ContextBuilder
  │   → DeepSeekGenerator
  ├─ 订单：OrderChatHandler
  │   → MCP Client → mcp_servers/order_server.py
  │   → DeepSeekGenerator.generate_order()
  └─ 闲鱼商品：
      → ItemContextResolver → MCP get_item_info
      → ItemFactResponder 或 XianyuKnowledgeResponder
      → scoped RAG / DeepSeek
→ Response
```

## P0

**未发现当前代码快照中可确认的 P0。**

## P1

### 1. 按 README/.env.example 的默认启动方式，RAG 核心链路不可用

- **文件**：`config/settings.py:25-26`，`.env.example:12-13`，`app/services/embedding_service.py:23-29`，`app/retrieval/reranker.py:66-73`
- **当前行为**：示例配置将 `EMBEDDING_MODEL_PATH` 和 `RERANKER_MODEL_PATH` 留空；两者实例化时都会抛 `FileNotFoundError`。
- **触发条件**：新环境按 README 执行 `copy .env.example .env` 后启动并发起普通 RAG 或闲鱼 RAG 请求。
- **实际影响**：服务可能启动，但核心检索无法完成；普通 RAG 被降级为“知识库服务不可用”，闲鱼知识问答转人工。
- **证据**：`EmbeddingService` 明确拒绝空路径；`Reranker._load_model()` 同样拒绝空路径。
- **建议方向**：README 必须将本地模型目录设为启动前置条件；启动阶段加入 readiness/health 检查，明确区分“HTTP 服务启动”与“RAG 可用”。
- **置信度**：高。

### 2. Qdrant 检索没有按 `corpus_id` 过滤，遗留向量可能进入当前回答

- **文件**：`app/infrastructure/qdrant.py:54-65, 125-153`，`app/retrieval/vector_search.py:28-48`
- **当前行为**：写入时每个 point 有 `corpus_id`；清理旧数据时也只删除“当前 corpus_id”的旧 point。但 `VectorSearch.search()` 未使用 `corpus_id` 过滤。
- **触发条件**：同一 collection 曾写入不同 `corpus_id`、迁移过知识库、手动写入过历史数据，或旧数据未被清理。
- **实际影响**：Dense Search 可能召回不属于当前知识库的内容；随后 RRF、Reranker 和 LLM 都会把它视为正常证据，造成错误回答和错误 sources。
- **证据**：`QdrantStore` 已设计 `corpus_id`，但读取端没有消费该隔离字段。
- **建议方向**：普通 RAG 查询也传入 `Filter(corpus_id=...)`；ingestion manifest 同时记录并校验 collection/corpus 标识。
- **置信度**：高。

## P2

### 3. 闲鱼渠道的 fail-closed 分支引用未定义变量

- **文件**：`app/channels/xianyu/action_mapper.py:51-53`
- **当前行为**：当 `/chat` 返回无 `action`、但 `can_answer=False` 时，代码访问未定义的 `clarify`。
- **触发条件**：普通 RAG/MCP 异常或旧格式响应返回 `can_answer=False`，且未携带 `action`。
- **实际影响**：抛出 `NameError`。上层 worker 会捕获广义异常并转人工，因此不会直接向买家发送错误回答，但会丢失原始失败原因，产生额外异常日志和不稳定渠道行为。
- **证据**：`clarify` 在此文件无定义；`MappedAction.action` 的合法值也不包含 `clarify`。
- **建议方向**：该分支直接返回 `MappedAction("human_handoff", reason=reason)`；补一个无 action 的 `can_answer=False` 回归测试。
- **置信度**：高。

### 4. 会话锁超时会直接变成 HTTP 500，而不是稳定的客服降级响应

- **文件**：`app/services/chat_service.py:120-125`，`app/services/session_manager.py:132-138, 145-150`
- **当前行为**：同一 `chat_id` 的请求会串行；等待超时后 `SessionManager` 抛 `TimeoutError`，`chat_async()` 和 API 层均未转换它。
- **触发条件**：同一会话的上一请求卡在 MCP、模型或 RAG，第二个请求等待超过 `SESSION_LOCK_TIMEOUT_SECONDS`（默认 60 秒）。
- **实际影响**：客户端收到 500，而不是可处理的“请求处理中/稍后重试”；闲鱼 worker 则进入通用异常转人工，诊断原因不准确。
- **证据**：锁的异常路径无 API 级处理。
- **建议方向**：为锁超时定义稳定错误响应或 HTTP 409/503；同时明确 MCP、LLM、RAG 的端到端超时预算。
- **置信度**：高。

### 5. 单订单查询没有返回 `mcp_result`，API schema 与实际能力不一致

- **文件**：`app/services/order_chat_handler.py:94-121`，`app/api/chat.py:59-66, 92-95`
- **当前行为**：`Response` 定义了 `mcp_result: OrderResponse`，但普通订单分支 `order()` 查询成功后只返回文本答案；只有 `combined()` 返回 `mcp_result`。
- **触发条件**：调用“查订单 TEST1001”这类纯订单查询。
- **实际影响**：前端无法稳定渲染订单状态、物流状态和运单号卡片，只能解析 LLM 文本；这也使订单事实缺少直接可验证的 API 输出。
- **证据**：`order()` 最终仅返回 `non_rag_response(...)`。
- **建议方向**：纯订单成功结果也返回经过 MCP schema 验证后的 `mcp_result`；LLM 文本只作为展示层补充。
- **置信度**：高。

### 6. 闲鱼处理内部 reason 被 API 响应模型丢弃，渠道人工接管缺少具体原因

- **文件**：`app/services/xianyu/responses.py:137-167`，`app/api/chat.py:82-100`，`app/channels/xianyu/action_mapper.py:36`
- **当前行为**：闲鱼 handoff 会填入 `reason`，但 `Response` 没有 `reason` 字段；Pydantic 默认忽略额外字段。渠道 HTTP 客户端收到的 payload 因而只有默认 `chat_handoff`。
- **触发条件**：商品不存在、商品证据不完整、事实冲突、RAG 证据不足等任一 handoff。
- **实际影响**：渠道数据库、企业微信通知和日志无法区分真实失败原因，排查与统计困难。
- **证据**：响应模型不声明 `reason`，但 mapper 依赖它。
- **建议方向**：在内部渠道契约中显式保留受控的 `reason_code`；不要暴露实现细节给普通 Web 用户。
- **置信度**：高。

## P3

### 7. `ContextBuilder` 接受没有有效 source/index 的上下文

- **文件**：`app/rag/context_builder.py:44-54`
- **当前行为**：只要 content 非空，就可进入 LLM context；source 可为空字符串，index 可为 `None`。
- **触发条件**：未来替换检索器、手工写 Qdrant payload、或注入不完整候选结果。
- **实际影响**：模型可能基于无法追溯的文本回答，返回给用户的 sources 与正文不一致。
- **证据**：`ContextBuilder` 未对 source/index 做有效性过滤；闲鱼路径已有 `valid_knowledge_sources()`，两处标准不一致。
- **建议方向**：普通 RAG 与闲鱼 RAG 统一证据有效性校验。
- **置信度**：中。

# 当前最严重的 5 个问题

1. 默认配置缺少 embedding/reranker 模型路径，核心 RAG 不可用。
2. Qdrant 未按 `corpus_id` 检索隔离，可能召回旧知识。
3. 闲鱼 `action_mapper` 的未定义变量导致异常分支失效。
4. 会话锁超时直接造成 HTTP 500。
5. 单订单查询丢失结构化 MCP 数据，前端只能依赖 LLM 文本。

# 当前最容易出错的调用链

```text
同 chat_id 并发请求
→ 前一请求卡在 RAG / LLM / MCP
→ SessionManager 等锁超时
→ TimeoutError 未转换
→ /chat 返回 500
→ 闲鱼 worker 捕获通用异常
→ 人工接管，但原因被泛化
```

# 已实现正确、不建议修改的部分

- RRF 按排名而非直接混合 BM25/Dense 原始分数，且以 `(source, chunk_index)` 去重。
- `Reranker` 对空候选、分数数量不一致和非有限分数有防护。
- `AnswerReliability` 校验 reranker 阈值与模型目录名的一致性，避免明显的模型/阈值错配。
- `RAGPipeline` 对上下文做分数窗口、chunk identity、内容归一化去重。
- 闲鱼 scoped corpus 同时对 Dense 和 BM25 使用商品范围过滤。
- MCP 客户端有读超时、结构化字段校验、连接上下文关闭。
- `SessionManager` 的 SQLite 锁释放使用 `finally`，会话隔离设计正确。
- `/chat` 对空 query、空 chat_id、额外字段已有 API 边界校验。

# 建议修复顺序

1. 配置可运行性：模型路径、启动 readiness、文档。
2. Qdrant `corpus_id` 查询过滤与数据完整性校验。
3. 修复 `action_mapper` 未定义变量，并增加该异常路径测试。
4. 统一会话锁超时和上游调用超时的降级策略。
5. 固化 `/chat` 的订单、handoff、reason_code 响应契约。

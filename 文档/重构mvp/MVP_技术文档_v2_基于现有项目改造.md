# 多平台 AI 客服底座 MVP 技术文档
> 版本：V2  
> 实施方式：基于现有仓库渐进式重构，不从零重写。

# 1. 技术目标

这次不是“重建项目”，而是“收缩现有架构”。

当前已有模块继续作为基础：

```text
app/api/chat.py
app/services/chat_service.py
app/services/session_manager.py
app/services/intent_router.py
app/services/query_planner.py
app/services/order_router.py
app/services/rag_service.py
app/services/order_chat_handler.py
app/services/mcp_service.py
app/services/xianyu/*
app/channels/xianyu/*
```

本次做法：

```text
先复用
→ 再包统一接口
→ 再迁移调用
→ 验收
→ 最后删除重复逻辑
```

而不是：

```text
先删掉旧代码
→ 重写全部功能
```

对应需求：R00—R09。

---

# 2. 现有模块与目标职责对应

| 现有模块 | MVP 处理方式 | 目标职责 |
|---|---|---|
| `app/api/chat.py` | 保留 | API 入口 |
| `chat_service.py` | 重点收缩 | 只编排主流程 |
| `session_manager.py` | 保留内部实现 | 统一会话读写 |
| `intent_router.py` | 先复用 | Planner 内部能力 |
| `query_planner.py` | 先复用 | Planner 内部能力 |
| `order_router.py` | 先复用后收敛 | Planner 的订单判断 |
| `rag_service.py` | 保留核心算法 | KnowledgeService 底层 |
| `order_chat_handler.py` | 复用并简化 | Order Handler |
| `mcp_service.py` | MVP 保留 | 订单/商品 MCP 封装 |
| `ProductAgent` | 复用 | Product Handler 核心 |
| `PriceAgent` | 复用 | Price Handler 核心 |
| `ServiceAgent` | 复用 | Service Handler 核心 |
| `expert_orchestrator.py` | 先复用执行能力 | 后续逐步收敛到统一 Executor |
| `stage3_worker.py` | 修改自动 handoff | 平台发送边界 |
| `action_mapper.py` | 修改动作映射 | 不再自动 HUMAN |
| `ChannelStore` | 保留 | 手动 AUTO/HUMAN 控制 |

---

# S0：先建立重构安全线

对应需求：R00、R09。

在改任何主流程前，先固定现有可工作的行为。

至少保留以下回归测试：

```text
1. /chat 普通知识问答
2. 商品事实问答
3. 闲鱼单问题
4. 闲鱼多问题
5. 订单 TEST1001 查询
6. Session 上下文
7. 闲鱼真实发送
```

目的：

> 后面每迁移一步，都能判断“只是架构变简单”，而不是“功能被改没了”。

### S0 验收

当前主分支/开发分支可以跑通基线测试，并记录结果。

---

# S1：在现有项目中增加统一数据合同

对应需求：R01、R03、R06、R07。

不要先移动大量文件。

建议先新增：

```text
app/services/chat_contracts.py
```

或：

```text
app/chat/contracts.py
```

定义：

## ChatMessage

```python
class ChatMessage:
    platform: str
    account_id: str
    chat_id: str
    buyer_id: str
    item_id: str | None
    text: str
```

## SessionContext

```python
class SessionContext:
    history: list
    current_item_id: str | None
    current_order_id: str | None
    last_task_type: str | None
    negotiation: dict
```

## Task

```python
class Task:
    task_id: str
    task_type: str
    query: str
```

`task_type`：

```text
product
price
service
order
```

## TaskResult

```python
class TaskResult:
    task_id: str
    status: str
    answer: str
    sources: list
    reason: str | None
```

## ChatResponse

```python
class ChatResponse:
    action: str
    answer: str
    results: list[TaskResult]
```

### S1 验收

现有闲鱼消息可以先转换成 `ChatMessage`，但底层收发逻辑不变。

---

# S2：给现有 SessionManager 增加统一入口

对应需求：R06。

不重写 `session_manager.py`。

在现有实现上增加：

```python
load(chat_id) -> SessionContext
```

内部继续复用：

```text
get_or_create
read_context
get_current_item_id
get_xianyu_context
```

再增加：

```python
save_turn(...)
```

内部继续复用：

```text
append_turn
set_current_item_id
update_xianyu_context
```

这样先把复杂性藏到 SessionManager 内部。

### 议价状态

在现有 state 中增加或映射：

```python
negotiation = {
    "item_id": None,
    "round": 0,
    "last_ai_offer": None,
    "last_buyer_offer": None,
}
```

### S2 验收

`ChatService` 不再需要连续调用多组 Session 方法来拼上下文。

---

# S3：建立统一 Planner，但先复用旧 Router

对应需求：R03。

不要立即删除：

```text
intent_router.py
query_planner.py
order_router.py
```

新增统一：

```text
Planner
```

第一版内部直接复用现有能力：

```text
Planner
 ├─ IntentRouter
 ├─ build_question_plan()
 ├─ build_expert_plan()
 └─ route_query()
```

但是对外只返回：

```text
Task[]
```

例如：

```text
这个修过吗？最低多少？多久发货？
```

输出：

```text
Task(product)
Task(price)
Task(service)
```

### 第二阶段再做

等 Planner 测试稳定后，再找重复规则：

```text
IntentRouter
QueryPlanner
OrderRouter
```

逐步合并。

### S3 验收

`ChatService` 只能调用：

```python
tasks = planner.plan(message, context)
```

不能再自己分别调用多个 Router。

---

# S4：把现有专家能力接入统一 Executor

对应需求：R04。

第一阶段不重新写 Product / Price / Service。

## Product

优先复用：

```text
ProductAgent
ItemFactResponder
XianyuKnowledgeResponder
```

包装成统一：

```python
handle(task, message, context) -> TaskResult
```

## Price

复用现有 `PriceAgent`：

```text
标价
底价保护
运费条件
授权优惠
```

MVP 先把返回结果适配成 `TaskResult`。

完整阶梯议价下一阶段再修改算法。

## Service

复用现有 `ServiceAgent` 和知识检索。

## Order

复用：

```text
OrderChatHandler
MCPService
order_mcp_client
```

第一版先保证订单查询功能不退化。

### S4 验收

同一个 `TaskExecutor` 可以根据 `task_type` 调用现有实现并返回统一 `TaskResult`。

---

# S5：在现有 RAGService 外增加 KnowledgeService

对应需求：R05。

不修改现有：

```text
BM25
Qdrant
HybridSearch
Reranker
RAGPipeline
Answerability
```

先新增：

```python
class KnowledgeService:
    def search(
        self,
        query,
        scope,
        platform=None,
        product_model=None,
        item_id=None,
    ):
        ...
```

内部继续调用现有：

```python
RAGService.prepare(...)
```

或现有 Xianyu scoped RAG。

KnowledgeService 负责统一：

```text
merchant
product
platform
item
```

四类 scope。

### 重要

第一版可以继续使用现有两个 collection / corpus。

不要为了“架构漂亮”先重建 Qdrant 数据。

先把接口统一，数据迁移以后再做。

### S5 验收

Product 和 Service 不直接关心 BM25/Qdrant/Reranker，只调用 KnowledgeService。

---

# S6：逐步收缩 ChatService

对应需求：R02。

这是核心重构步骤。

不要一次性重写整个文件。

建议按顺序迁移：

## 第一步

把 Session 读取改成：

```python
context = session_manager.load(chat_id)
```

## 第二步

把：

```text
intent_router
build_question_plan
route_query
_should_use_xianyu_experts
```

从 ChatService 入口移到 Planner。

## 第三步

把具体分支执行移入 Executor / Handler。

## 第四步

把结果合并放入 ResultMerger。

最终 ChatService 目标：

```python
async def handle(message):
    context = session_manager.load(message.chat_id)

    tasks = planner.plan(message, context)

    results = await executor.execute(
        tasks,
        message,
        context,
    )

    response = merger.merge(results)

    session_manager.save_turn(
        message,
        response,
    )

    return response
```

### S6 验收

`ChatService` 不再包含大量业务关键词和特殊 route 分支。

---

# S7：删除 `rag_mcp` 作为长期特殊路径

对应需求：R03、R04。

现有 `OrderChatHandler.combined()` 先保留，避免第一天破坏订单能力。

新 Planner 稳定后：

```text
TEST1001 怎么还没到？一般多久到？
```

拆成：

```text
order
service
```

分别执行：

```text
Order → MCP
Service → KnowledgeService
```

最后交给 ResultMerger。

等该流程验收通过，再让：

```text
rag_mcp
combined()
```

退出主链路。

### S7 验收

组合问题不再依赖单独的 `rag_mcp` route。

---

# S8：修改自动 handoff，不删除人工控制

对应需求：R08。

重点修改现有：

```text
app/channels/xianyu/action_mapper.py
app/channels/xianyu/stage3_worker.py
app/services/xianyu/responses.py
app/services/xianyu/expert_orchestrator.py
```

## action_mapper

修改：

```text
clarify
→ clarify
```

不能：

```text
clarify
→ human_handoff
```

`can_answer=False` 也不能自动等于 HUMAN。

## stage3_worker

自动流程禁止调用：

```python
store.handoff(...)
```

普通失败：

```text
记录错误
或发送安全提示
但保持 AUTO
```

## ChannelStore

以下能力保留：

```text
set_session_mode
takeover
release
handoff 相关历史兼容
```

但新自动主链路不再调用 `handoff()`。

### S8 验收

模拟：

- RAG 无结果；
- 缺商品字段；
- Agent 失败；
- 模型超时；
- `/chat` 异常。

数据库必须保持：

```text
mode = AUTO
```

手动 takeover 才能变 HUMAN。

---

# S9：ResultMerger 支持部分成功

对应需求：R07。

不要直接复用现有：

```text
任一 ExpertResult 失败
→ 整轮 handoff
```

新增统一合并逻辑。

示例：

```text
product = answered
price = answered
service = unavailable
```

输出：

```text
商品没有维修记录。
目前价格可以小刀。
周日是否发货暂时无法确认。
```

### S9 验收

至少测试：

```text
全部成功
2 成功 + 1 失败
1 成功 + 2 失败
全部 unavailable
```

都不能自动转 HUMAN。

---

# S10：接回现有闲鱼真实链路

对应需求：R01、R09。

复用现有：

```text
app/channels/xianyu/*
```

只修改核心边界：

```text
原始闲鱼消息
→ ChatMessage
→ 新 ChatService 主链路
→ ChatResponse
→ 现有发送器
```

不要重写 Cookie、WebSocket、Sender、Store 等已打通功能。

### S10 验收

真实闲鱼：

```text
收消息
→ 新主链路
→ 回复
```

完整跑通。

---

# 3. 删除旧逻辑的顺序

只有新链路验收完成后，才允许清理旧代码。

顺序：

```text
1. 先迁移调用
2. 验收
3. 搜索旧调用是否仍存在
4. 确认无引用
5. 再删除重复实现
```

优先考虑后续清理：

```text
ChatService 中重复判断
旧的特殊 route
重复 QueryPlanner 判断
不再使用的自动 handoff 分支
```

暂时不要急着删除：

```text
RAGService
SessionManager
MCPService
OrderChatHandler
现有 Agent
ChannelStore
```

---

# 4. 最终验收矩阵

| 需求 | 技术步骤 | 验收 |
|---|---|---|
| R00 基于现有项目改造 | S0—S10 | 旧能力复用，无平行重写 |
| R01 统一平台消息 | S1、S10 | 闲鱼 → ChatMessage |
| R02 收缩 ChatService | S6 | load → plan → execute → merge → save |
| R03 统一 Planner | S3、S7 | 只对外输出 Task[] |
| R04 复用四类能力 | S4 | 现有 Agent/MCP 继续工作 |
| R05 RAG 知识层 | S5 | KnowledgeService 包装现有 RAG |
| R06 Session | S2 | 保留 SQLite，仅简化接口 |
| R07 部分成功 | S9 | 子任务失败不拖垮整轮 |
| R08 取消自动 HUMAN | S8 | 只有手动 takeover |
| R09 闲鱼兼容 | S10 | 真实闲鱼链路继续工作 |

---

# 5. 本次 MVP 最重要的工程原则

这次重构成功的标准不是“新建了多少文件”。

而是：

```text
功能没丢
代码更少绕路
主流程更清楚
旧能力被复用
特殊分支减少
```

最终应该做到：

```text
现有项目
   ↓
逐步收拢
   ↓
形成统一客服底座
```

而不是：

```text
现有项目
+
另一套新架构
=
两个系统同时维护
```

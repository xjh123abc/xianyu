# 多平台 AI 客服底座 MVP 需求文档
> 版本：V2  
> 改造原则：基于现有项目渐进式重构，不从零重写。

## 1. 项目背景

当前项目已经具备以下能力：

- `/chat` API；
- `ChatService` 主流程；
- `SessionManager` 会话；
- `IntentRouter` / `QueryPlanner`；
- RAG 检索、混合召回、Reranker、Answerability；
- 商品事实处理；
- Product / Price / Service 专家；
- 订单 MCP；
- 闲鱼消息接收与发送；
- AUTO / HUMAN 会话控制。

因此，本次 MVP **不是重新搭建一套系统**。

本次目标是：

> 在现有代码可以工作的基础上，把过于分散的判断和调用关系收拢成一条清晰主链路，同时保留已经验证可用的 RAG、MCP、Session、闲鱼收发等能力。

---

# 2. MVP 总目标

建立一个可继续扩展到多个电商平台的 AI 客服核心框架。

第一阶段继续以闲鱼作为真实接入平台，但核心业务层不再绑定闲鱼。

最终主链路固定为：

```text
闲鱼消息
  ↓
现有闲鱼 Adapter / Worker
  ↓
统一 ChatMessage
  ↓
ChatService
  ↓
SessionManager
  ↓
Planner
  ↓
Product / Price / Service / Order
  ↓
KnowledgeService / 商品数据 / MCP
  ↓
ResultMerger
  ↓
统一 ChatResponse
  ↓
现有闲鱼发送链路
```

---

# 3. 本次改造的硬性原则

## R00：必须基于现有项目改造

本次 MVP 必须遵守：

1. 不新建第二套平行客服系统；
2. 不推翻已经可用的 RAG；
3. 不推翻现有 Session 存储；
4. 不重写闲鱼底层收发链路；
5. 不一次性删除旧模块；
6. 先建立统一入口，再逐步把旧逻辑迁移进去；
7. 每迁移一个模块，都必须先通过现有功能回归测试。

允许：

```text
旧模块
→ 包一层统一接口
→ 新主链路开始调用
→ 验收通过
→ 再删除重复旧逻辑
```

禁止：

```text
先大规模删除旧代码
→ 再重新实现
→ 最后才发现原有功能丢失
```

---

# 4. 功能需求

## R01：统一平台消息

复用现有闲鱼接收链路。

闲鱼消息进入核心业务前，转换成统一结构：

```text
platform
account_id
chat_id
buyer_id
item_id
text
```

以后新增淘宝、拼多多时，只增加新的 Adapter。

核心 `ChatService` 不直接依赖闲鱼原始消息格式。

---

## R02：收缩现有 ChatService

现有 `ChatService` 保留，不重新造一个平行版本。

逐步把它收缩到只负责：

1. 读取 Session；
2. 调用 Planner；
3. 执行任务；
4. 合并结果；
5. 保存 Session；
6. 返回响应。

现有散落在 `ChatService` 中的：

- 商品字段判断；
- 闲鱼专家选择；
- 订单路由；
- RAG / MCP 特殊组合；
- 自动 handoff 判断；

逐步下沉到对应模块。

---

## R03：把现有 IntentRouter / QueryPlanner 收拢成统一 Planner

MVP 不要求立即删除：

- `intent_router.py`
- `query_planner.py`
- `order_router.py`

第一阶段允许 Planner 内部继续调用这些已有能力。

但对 `ChatService` 只暴露一个统一结果：

```text
product
price
service
order
```

一个用户问题可以生成多个任务。

例如：

```text
这个修过吗？最低多少？多久发货？
```

输出：

```text
product
price
service
```

后续验收稳定后，再逐步删除重复判断。

---

## R04：复用现有四类业务能力

### Product

优先复用现有：

- `ProductAgent`
- `ItemFactResponder`
- 商品数据读取能力。

负责：

- 当前商品事实；
- 产品/型号专业知识。

### Price

复用现有 `PriceAgent` 的：

- 标价读取；
- 底价保护；
- 运费条件；
- 卖家授权价格规则。

MVP 暂不要求完成完整阶梯议价，但必须预留议价状态。

### Service

复用现有：

- `ServiceAgent`
- `XianyuKnowledgeResponder`
- 商家规则检索能力。

### Order

复用现有：

- `OrderChatHandler`
- `MCPService`
- `order_mcp_client`。

MVP 先保留可工作的订单查询能力，再逐步去掉 `rag_mcp` 特殊分支。

---

## R05：把现有 RAG 收敛成统一 KnowledgeService

现有 RAG 内部实现继续保留：

```text
BM25
Qdrant
HybridSearch
Reranker
Answerability
ContextBuilder
```

本阶段不重新实现这些算法。

新增统一知识层入口 `KnowledgeService`，由它包装现有 `RAGService`。

知识层只负责：

> 从可信知识库中检索可供业务模块使用的证据。

MVP 支持四类知识：

- 商家通用规则；
- 产品/型号知识；
- 平台规则；
- 商品长文本说明。

以下内容不放入 RAG 作为主要事实源：

- 当前价格；
- 库存；
- 实时订单；
- 实时物流；
- 当前议价轮次；
- 当前 Session 状态。

---

## R06：复用现有 SessionManager，但简化调用方式

保留现有：

- SQLite；
- TTL；
- Session Lock；
- 历史消息；
- 当前商品；
- 当前订单。

不重新写存储层。

只在外部增加更清晰的统一调用：

```text
load()
save_turn()
```

并预留议价状态：

```text
item_id
round
last_ai_offer
last_buyer_offer
```

---

## R07：多任务允许部分成功

修改现有专家汇总策略。

不得再出现：

```text
3 个子任务
其中 1 个失败
→ 整轮直接失败
```

应改成：

```text
已确认部分正常回答
+
无法确认部分明确说明
```

例如：

```text
没有维修记录。
包含镜头。
周日是否发货目前没有明确记录。
```

---

## R08：取消自动人工接管

保留现有：

```text
AUTO / HUMAN
takeover()
release()
```

但修改规则：

> 只有卖家主动执行 takeover，才能把会话改成 HUMAN。

以下情况都不得自动切 HUMAN：

- RAG 无结果；
- 商品资料不足；
- Planner 子任务失败；
- 需要澄清；
- 模型超时；
- `/chat` 异常；
- 一个 Agent 回答失败。

自动流程只允许：

```text
reply
clarify
ignore
error
```

---

## R09：保持现有闲鱼链路可用

本次架构重构不能破坏已经打通的：

```text
闲鱼收消息
→ /chat
→ 生成答案
→ 闲鱼发送
```

改造时优先保持现有 API 和发送边界兼容。

---

# 5. MVP 暂不实现

本阶段不做：

- 第二个平台真实接入；
- Web Search；
- 完整阶梯式议价；
- 长期记忆；
- 大规模文件目录迁移；
- 全量删除旧 Router；
- 新增更多 Agent；
- 重写现有 RAG 算法。

这些功能只留接口或后续规划。

---

# 6. 验收标准

## A00：确认是“改造”而不是“重写”

验收时必须确认：

- 原有 `/chat` 入口仍可用；
- 原有闲鱼收发链路仍可用；
- 原有 RAG 检索能力仍被复用；
- 原有订单 MCP 仍被复用；
- 原有 Session 数据仍可继续使用；
- 没有出现第二套并行 ChatService。

---

## A01：商品事实

输入：

```text
这个相机带镜头吗？
```

要求：

```text
Planner
→ Product
→ 现有商品事实能力
→ 正常回复
```

---

## A02：产品知识

输入：

```text
Canon FTb 怎么测光？
```

要求：

```text
Planner
→ Product
→ KnowledgeService
→ 现有 RAG
→ 正常回复
```

---

## A03：服务规则

输入：

```text
一般多久发货？
```

要求：

```text
Planner
→ Service
→ KnowledgeService
→ 现有 RAG
```

---

## A04：订单

输入：

```text
TEST1001 到哪了？
```

要求：

```text
Planner
→ Order
→ 现有 MCP
→ 正常回复
```

---

## A05：多问题

输入：

```text
这个修过吗？最低多少？多久发货？
```

至少拆出：

```text
product
price
service
```

并最终合并成一条回复。

---

## A06：部分失败

三个任务中一个失败：

- 成功结果必须保留；
- 失败部分不得编造；
- Session 必须继续是 `AUTO`。

---

## A07：取消自动 HUMAN

分别模拟：

- RAG 无结果；
- 商品字段缺失；
- Agent 失败；
- 模型超时；
- `/chat` 异常。

验收：

```text
channel_sessions.mode = AUTO
```

只有手动 takeover 后才允许：

```text
mode = HUMAN
```

---

## A08：ChatService 主流程清晰

最终阅读 `ChatService` 时，核心结构应接近：

```text
load
→ plan
→ execute
→ merge
→ save
→ return
```

---

# 7. MVP 完成定义

同时满足：

1. 基于现有项目完成渐进式改造；
2. 没有重写已稳定能力；
3. 主流程统一；
4. Planner 成为统一任务入口；
5. Product / Price / Service / Order 职责清楚；
6. RAG 成为统一 KnowledgeService；
7. 自动 HUMAN 被取消；
8. 多问题允许部分成功；
9. 闲鱼真实链路仍能完整工作。

才算本次 MVP 完成。

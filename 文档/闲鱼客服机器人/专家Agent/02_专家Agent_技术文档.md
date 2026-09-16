# 闲鱼专家 Agent 改造技术文档

版本：V2 · 仓库结构版  
代码基线：`xjh123abc/xianyu` / `xianyu-dev` / `385ee01e8f9007ec34530c7a06ce9ec55ece6b9c`  
配套文档：[需求文档](01_专家Agent_需求文档.md)

> 本文是实现计划；标记“新增”的文件、类和方法尚不存在。本次只上传文档，未实现或运行下述新功能。

## 1. 接口放在哪里

**不新增HTTP接口。** 继续使用 `app/api/chat.py` 中的 `POST /chat`：

```json
{
  "query": "还在吗？修过没有？不包邮最低多少？",
  "chat_id": "expert_test_001",
  "item_id": "CANON_FTB_001"
}
```

业务接入点是 `app/services/chat_service.py`：

1. 在 `ChatService.__init__()` 注入并组装新增的 `XianyuExpertOrchestrator`。
2. 在 `_chat_async_locked()` 的闲鱼回答分支调用编排层，继续使用已经解析的商品和会话。
3. 保留 `route_query()` 的订单分流；订单查询、`rag_mcp`、普通电商RAG和原有订单规则追问不被专家截走。
4. 商品不存在或身份冲突，保留 `ItemContextResolver.resolve()` 的提前返回，不由专家猜商品。
5. `append_turn()` 仍由 `ChatService` 调用一次，专家不单独写聊天历史。

```text
app/api/chat.py : chat()
  → ChatService.chat_async()
    → 现有 session_lock()
      → _chat_async_locked()
        → 商品解析与原有订单／普通RAG分流
        → 闲鱼分支：XianyuExpertOrchestrator.handle()
          → build_expert_plan()
          → ProductAgent / PriceAgent / ServiceAgent
          → 检查各任务结果，合并为一条回复
        → 原有 append_turn()
  → 原有 Response
  → S3的 map_chat_response()
  → 正常发送或 _handoff()
```

专家代码放在 `app/services/xianyu/`，不要放进负责收发的 `app/channels/xianyu/`，也不要修改 `.runtime/xianyu-template` 来实现业务。专家不得反向调用 `/chat`。

## 2. 文件落点和内部接口

### 2.1 新增文件

```text
app/services/xianyu/
├─ expert_orchestrator.py        # 调度、检查、合并，不负责发送
└─ experts/
   ├─ __init__.py
   ├─ contracts.py               # 任务、共享上下文、专家结果
   ├─ price_agent.py             # 价格政策解析、金额计算和报价判断
   ├─ product_agent.py           # 单品事实与型号知识
   └─ service_agent.py           # 发货、售后等卖家服务

app/generation/
└─ xianyu_expert_prompt.py       # 规划、商品、服务的专用提示词

tests/xianyu/
├─ test_xianyu_price_agent.py
├─ test_xianyu_expert_agents.py
├─ test_xianyu_expert_plan.py
└─ test_xianyu_expert_orchestrator.py

tests/api/
└─ test_xianyu_expert_contract.py
```

### 2.2 修改／复用现有文件

| 文件 | 具体操作 |
|---|---|
| `app/services/chat_service.py` | 组装编排层，替换闲鱼回答选择逻辑；保留会话、商品解析、订单分流 |
| `app/services/query_planner.py` | 新增 `build_expert_plan()`，返回完整任务列表 |
| `app/services/intent_router.py` | 保留分类；增加可关闭模型兜底的规则检查，避免复合问题重复分类 |
| `app/services/xianyu/item_fact_responder.py` | 抽出PRICE/BARGAIN价格逻辑；保留商品／服务事实读取 |
| `app/services/xianyu/knowledge_responder.py` | 抽出公共证据准备方法供专家使用，不再对整条复合消息直接提前回答 |
| `app/services/xianyu/responses.py` | 复用 `reply()`、`handoff()`、`common_handoff()` 和固定话术检查 |
| `app/generation/deepseek.py` | 新增规划与专家生成方法，复用现有模型配置和调用封装 |
| `app/services/session_manager.py` | 在现有 `state_json` 存商品级追问上下文，不建新数据库 |
| `app/api/chat.py` | 给 `Response` 增加可选 `reason` 字段；其他请求／响应字段保持兼容 |
| `app/channels/xianyu/action_mapper.py` | 修复不可回答分支中的未定义变量，保持接管语义 |
| `data/xianyu/items.json` | 只澄清Canon两条价格政策文本，保留其他商品事实 |

`ItemService`、商品MCP、`RAGService`、S3工作器、`ChannelStore` 和企业微信模块直接复用，不在本次另建实现。

### 2.3 专家共用一个契约

在 `experts/contracts.py` 定义：

| 契约 | 必须携带的信息 |
|---|---|
| `ExpertTask` | `task_id`、专家名、原问题片段、规范化问题、知识范围、交易条件、依赖任务ID |
| `ExpertContext` | 当前完整问题、已确认商品或None、同会话历史、商品级追问上下文 |
| `ExpertResult` | `task_id`、专家名、状态、答案片段、来源、缺失字段、内部原因 |

专家名仅为 `product / price / service`；知识范围区分 `item_fact / model_knowledge / seller_rule / greeting`；结果状态第一版仅为 `answered / handoff`。

统一内部接口如下，不是新HTTP接口：

```python
async def run(
    self,
    tasks: list[ExpertTask],
    context: ExpertContext,
) -> list[ExpertResult]:
    ...
```

一个任务必须对应一个结果；同一专家批量处理自己的任务。简单招呼不要求证据，其他事实必须关联实际商品字段或检索片段。模型可以返回证据引用，但程序必须核对引用确实来自本轮输入；不能相信模型自行编出的来源。

金额另存整数分及运费条件，不能只埋在答案文本里。外部继续使用原有 `source/index` 格式，不暴露整套内部任务对象。

## 3. S1：固定基准，确认接入点

**要做什么：**

1. 从 `xianyu-dev` 建立实施分支，核对本文代码基线之后的变化。
2. 保存Canon商品快照，运行现有测试并记录结果。
3. 标出 `ChatService` 里的闲鱼回答入口和提前返回，避免新旧回答逻辑同时处理一条消息。

```powershell
python -m pytest tests/xianyu tests/api tests/orders tests/regression -q
```

本次阅读代码已发现三个需要在迁移时处理的具体位置：

- `XianyuKnowledgeResponder.handle_item()` 遇到BARGAIN直接返回价格结果。不能保留这个整条消息提前返回，否则“维修＋议价”仍可能只报价。
- `responses.handoff()` 已返回 `reason`，但API的 `Response` 没有这个字段。要验证原因穿过HTTP后仍保留。
- `map_chat_response()` 的无action且 `can_answer=False` 分支引用未定义的 `clarify`。先用测试复现，S5修复。

**产出：** 固定数据、基准测试记录、接入点。  
**完成条件：** 已区分原有失败与新增失败；本步不替换真实回答路径。本次文档编写没有运行这些测试，不能将上述静态检查当作测试通过记录。

## 4. S2：先迁移价格专家

### 4.1 沿用真实商品字段

当前数据链路为：

```text
data/xianyu/items.json
  → app/services/item_service.py 的 _validate_structured_facts()
  → MCP get_item_info → item["facts"]
  → ItemContextResolver.resolve()
  → PriceAgent
```

商品字段校验已保留 `negotiation`、`shipping.negotiation_policy`、`shipping.negotiation_express_policy`，MCP客户端透传 `facts`。本期不另加一套顶层 `pricing` 字段。

Canon的两条政策文本建议统一如下。这里只展示要修改的局部，不能覆盖整个商品对象：

```json
{
  "negotiation": "negotiable",
  "shipping": {
    "negotiation_policy": "累计最多小刀10元",
    "negotiation_express_policy": "不包邮商品价减20元，可叠加小刀"
  }
}
```

保留原有发货地、快递、包邮与发货时限等字段，`listed_price_cents=150000` 保持不变。

### 4.2 迁移计算，不保留两套价格规则

从 `ItemFactResponder` 迁移PRICE／BARGAIN分支及价格辅助能力到 `price_agent.py`。短期旧入口可委托新实现；不得复制后让两套规则长期并行。

解析为内部价格政策：原始标价、累计小刀上限、不包邮减免、是否叠加。使用整数分计算，输入小数金额沿用Decimal转换。

```text
包邮下限 = 150000 - 1000 = 149000 分
不包邮下限 = 150000 - 2000 - 1000 = 147000 分
```

只支持明确政策格式。“1480元／1490元”这样的参考成交价不能当折扣。无法解析、文本相互冲突、优惠超过标价时返回缺失／冲突，不让模型临时报价，也不静默把底价截为零。

买家问题区分为查标价、查最低价、明确出价、继续要求优惠：

- 查最低价：返回适用方案下限；比较方案就分别计算。
- 明确出价：达到授权范围时按买家出价回答，例如1495，不主动降到1490。
- 再优惠：结合本商品上一轮话题理解请求，再对原始标价核对累计优惠。
- 包邮条件不可替换。自提、拆卖等没有授权政策时不得直接同意。
- 附带“没修过才买”等条件时，依赖相关商品任务；条件未核实不得承诺成交。

价格专家还要检查商品在售状态和相关 `fact_conflicts`；已售出只说明状态，不接受购买。其他商品不能继承Canon的优惠；`negotiation=firm` 时小刀为零，未知政策不当作已授权。

**产出：** 最小契约、可独立测试的PriceAgent、唯一价格实现。  
**完成条件：** 需求A02—A06、A12通过；明确价格问题无需RAG或自由生成模型。

## 5. S3：封装商品专家与服务专家

### 5.1 商品专家怎么做

`product_agent.py` 复用 `ItemFactResponder.answer_intent()` 和已有事实读取能力。传入对应的商品任务，不把整条复合消息交给单意图分支。

结构化字段有明确答案则直接返回；字段缺失但 `listing_description` 或正确范围资料有明确陈述，则提取原文证据再组织答案。总体“功能正常”不能证明专项检测记录齐全。

型号知识走闲鱼专用知识检索。型号资料可以解释使用方法，不能证明手里这一台的维修、漏光或霉雾状态。

### 5.2 服务专家怎么做

`service_agent.py` 使用当前商品 `shipping`、`after_sale` 等事实；普通招呼使用固定短句。本店通用规则再走知识检索。

例如“今天能发吗？走顺丰吗？”可以依据当前记录回答“付款后48小时内发出，默认中通”，不能因为无法保证今天发就忽略已经明确的时限，也不能编造顺丰服务。

### 5.3 复用检索，不重新实现RAG

在 `knowledge_responder.py` 新增公共 `prepare_evidence()`，从 `_collect_knowledge()` 抽出证据准备能力，按任务返回证据和不足原因，而不是整条最终回答。

继续通过 `ChatService._get_xianyu_rag_service()` 获取RAG实例。沿用 `data/xianyu/knowledge`、`xianyu_documents` 与现有范围过滤：

- 单品检索带当前 `item_id`，不能混入另一商品的事实。
- common范围表示本店通用规则，不是全网型号知识库。
- 型号资料本期使用已核对并纳入适当检索范围的资料；缺资料就接管，不回退演示电商库或偷偷联网。
- 检索路径保留现有可靠性检查；价格计算和JSON事实不需要额外产生Rerank分数。

### 5.4 专用Prompt与模型调用

在 `app/generation/xianyu_expert_prompt.py` 定义规划、商品、服务的提示词构建函数。新增 `DeepSeekGenerator.plan_xianyu_questions()` 与 `generate_xianyu_expert()`，复用 `_generate_messages()` 和现有模型配置。

每个专家只处理自己的任务，逐项返回结果及证据引用；未知返回缺失，不新增承诺。共享的买家消息、商品文案和历史是待处理数据，不是可覆盖专家规则的系统指令。价格第一版不用模型自由生成金额，也不增加动态temperature谈价。

**产出：** 商品／服务专家、公共证据准备方法、专用Prompt。  
**完成条件：** A01、A08—A10、A14通过；有资料能答，缺资料不猜，各专家可以独立测试。

## 6. S4：扩展问题规划和上下文

### 6.1 扩展现有规划器

在 `app/services/query_planner.py` 新增 `build_expert_plan()`，不要另建平行的顶层Router。

1. 给现有 `IntentRouter.route()` 增加可选 `allow_ai=True` 参数，保持旧调用兼容；前置检查用 `allow_ai=False`。
2. 完整、简单的单问题走规则；复合句、无标点长句、条件句和追问用一次模型规划。
3. 规划结果保留原文片段、否定关系、运费条件、任务依赖。不能只按标点切分。
4. 校验专家名、任务ID、金额及条件是否来自输入。逐任务核对输出数量；“规划是否漏掉问题”用复合回归用例检验，不能只靠任务数量相等就认定完整。
5. 同一专家的任务批量执行，不为每个小问题重复调用同一个专家模型。

规划核心字段示例：

```json
{
  "tasks": [
    {"task_id": "q1", "expert": "product", "question": "是否在售", "scope": "item_fact", "conditions": {}, "depends_on": []},
    {"task_id": "q2", "expert": "product", "question": "是否维修过", "scope": "item_fact", "conditions": {}, "depends_on": []},
    {"task_id": "q3", "expert": "price", "question": "不包邮最低价", "scope": "item_fact", "conditions": {"shipping": "buyer_pays", "request_kind": "minimum"}, "depends_on": []}
  ]
}
```

“如果没修过，1470不包邮我就买”应让成交任务依赖维修任务；只询问最低价不代表买家已接受。

### 6.2 复用会话状态

沿用 `SessionManager.read_context()`。在已有 `state_json` 增加 `xianyu_context`：商品ID、最近价格话题、买家明确选择的运费方案；通过新增公开更新方法写入，不让专家调用私有 `_save()`。

旧会话读取时给新增键默认值，不改变订单字段。商品变化清空旧交易条件；比较方案不记录为选择。没有 `item_id` 的短追问，但会话已有当前商品时，也要进入专家上下文判断，不能落入普通电商RAG。

历史AI回复只能辅助理解话题，不能成为卖家新授权。当前 `append_turn()` 在HTTP返回前记录答案，不等于闲鱼已经发送，更不等于成交。

**产出：** 任务规划、商品级追问状态。  
**完成条件：** A07、A10—A13通过；不重复模型分类，不串商品，不用历史错误报价改底价。

## 7. S5：统一出口并接回主链路

### 7.1 编排层负责什么

在 `expert_orchestrator.py` 新增 `XianyuExpertOrchestrator.handle(query, *, item, history, session_state)`，返回原有响应字典。

处理顺序为：规划 → 按需执行专家与依赖任务 → 核对结果 → 合并。

- 全部可答：程序按问题顺序合并，去除重复招呼；价格使用计算结果，不再让总控模型改写。
- 任一必要任务缺失、冲突或失败：调用现有 `handoff()`／`common_handoff()`，整条只返回“稍等我看看”。
- 来源按真实任务结果合并去重；检索结果保留原始来源，不捏造统一置信度或平均不同任务的分数。
- 所有生成出口经过已有 `requires_human_handoff()` 等边界检查；模型自报 `answered` 不是事实依据。

同一专家先批量处理独立任务，依赖条件全部通过后才能给出成交承诺。第一版无需专家互相讨论，也不需要为此引入新框架。

### 7.2 修改 `ChatService`

在构造函数增加可选编排器注入，保留现有构造参数和测试替身，继续使用 `_get_generator()` 与 `_get_xianyu_rag_service()`。

闲鱼回答分支统一调用编排器，替代旧的简单／复杂／BARGAIN提前返回选择。无具体商品的本店规则咨询交服务专家；订单、普通RAG仍按原分支处理。

`needs_item` 的前置判断可以继续使用现有轻量规则；复合语义规划只在确定的闲鱼分支运行一次。商品解析失败提前接管，不再做一轮无意义的专家生成。

原来末尾的 `attach_intent_metadata()` 只处理单个意图，不能覆盖新复合结果。单任务保留原意图名；复合任务在已有字符串 `intent` 中返回 `MULTI`，`required_fields` 合并去重。对话仍只记一次。

切换后一个请求只能运行一套回答逻辑。迁移完成移除不再使用的旧提前返回和重复分支，不长期维护新旧两套价格与规划实现。

### 7.3 让接管原因穿过HTTP

在 `app/api/chat.py` 的 `Response` 新增：

```python
reason: str | None = None
```

`answer` 与 `reason` 分离：买家只看到等待话术；reason保存简短缺失项及已确认结果，不放密钥或原始异常堆栈。

```json
{
  "query": "测光对比过吗？不包邮最低多少？",
  "chat_id": "expert_test_002",
  "item_id": "CANON_FTB_001",
  "route": "xianyu",
  "action": "handoff",
  "answer": "稍等我看看",
  "can_answer": false,
  "next_step": "human_handoff",
  "reason": "缺少测光对比记录；已确认不包邮最低1470元"
}
```

`map_chat_response()` 继续读取reason，`XianyuStage3Worker._handoff()` 负责状态和通知。修复未定义的 `clarify` 分支；无action且不可回答、空回复和矛盾结果应稳定接管，不能把它们当正常答案发送。

**不要返回 `clarify` 并期待买家收到追问：当前mapper会将它转人工，本期不改变这个约定。** 保留S3的 `prepare_candidate()`、`claim_ready_delivery()` 等发送前状态检查，防止卖家接管后旧答案仍发送。

### 7.4 约束多次调用的等待时间

当前渠道默认等待30秒，DeepSeek默认超时60秒。接入多个专家时，整条请求应使用小于渠道等待的总处理预算，并将剩余时间传给实际模型网络调用，限制重试。

同步模型／检索调用不能长期阻塞事件循环。放入受限线程后，等待超时也不代表底层请求停止；迟到结果必须丢弃，由已有S3完成接管。不要只调大模型超时而忽略渠道已提前超时。

**产出：** 贯通 `/chat`、专家、HTTP响应和S3的链路。  
**完成条件：** A15—A18通过；一次最终决策、reason不丢失，专家不直接通知或发消息。

## 8. S6：测试后再启用

### 8.1 单元与契约测试

使用第2节的新测试文件，复用现有 `NoRag`、`Mock/AsyncMock`，隔离测试会话，不调用真实企业微信。

覆盖价格计算、商品事实、任务完整性、上下文及错误输出。API测试必须经过完整HTTP序列化；扩展S3测试检查接管状态、通知次数、重复消息与生成中人工接管。

旧测试里“必须调用一次 `generate_xianyu()`”是旧实现断言，可改为对应专家调用；不得删除“不乱答、不漏问、不无故检索”等业务断言。旧自提测试当前允许1490直接成交，本期按需求R03改为未授权条件接管，这是明确的行为变更，不是放宽测试。

```powershell
python -m pytest tests/xianyu tests/api tests/orders tests/regression -q
python -m pytest -q
```

### 8.2 固定批量问题

复用已有60条问题，补齐需求A01—A18。独立问题用独立chat_id，连续追问按组复用chat_id；保存答案、action、任务分配、缺失原因和耗时。

新旧比较使用同一商品快照。通过真实渠道测试时，上一个handoff会影响后续消息，使用隔离会话或组间手动恢复，不能自动释放真实人工会话。

### 8.3 小范围真实验证

使用受控测试会话验证正常报价、未知事实接管和生成期间手动接管。确认买家只收到一条消息，企业微信包含具体缺失原因后再扩大使用。

分别记录代码测试、真实模型抽测、真实渠道联调的结果。Mock测试通过不表示真实渠道已通过；未运行的项目必须标明。

**最终完成条件：** 需求A01—A18通过；原订单/RAG没有新增回归；真实自动发送只在受控验收通过后启用。

## 9. 提交顺序

```text
S1 固定基准与入口
→ S2 迁移价格专家
→ S3 商品／服务专家
→ S4 任务规划与上下文
→ S5 接回ChatService、HTTP和S3
→ S6 测试后启用
```

每一步单独提交，写明修改文件、测试命令、结果和未完成项。最后接线前不替换现有在线路径；接线失败回退对应提交并保留商品快照，不重置或覆盖远程分支。

## 代码依据

本设计核对了以下文件，路径均相对仓库根目录：

- [ChatService](../../../app/services/chat_service.py)、[现有意图路由](../../../app/services/intent_router.py)、[问题规划](../../../app/services/query_planner.py)。
- [商品字段校验](../../../app/services/item_service.py)、[商品解析](../../../app/services/xianyu/item_context_resolver.py)、[MCP返回字段](../../../app/infrastructure/order_mcp_client.py)。
- [事实／价格回答](../../../app/services/xianyu/item_fact_responder.py)、[知识处理](../../../app/services/xianyu/knowledge_responder.py)、[响应构造](../../../app/services/xianyu/responses.py)。
- [生成封装](../../../app/generation/deepseek.py)、[现有提示词](../../../app/generation/prompt.py)、[会话存储](../../../app/services/session_manager.py)、[配置](../../../config/settings.py)。
- [API响应模型](../../../app/api/chat.py)、[渠道映射](../../../app/channels/xianyu/action_mapper.py)、[S3工作器](../../../app/channels/xianyu/stage3_worker.py)、[复合消息测试](../../../tests/xianyu/test_xianyu_complex_buyer_messages.py)。

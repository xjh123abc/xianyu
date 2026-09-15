# AI 智能客服项目：数据库学习与落地指南

> 适用项目：基于 FastAPI + RAG + Qdrant + MCP + 闲鱼消息链路的 AI 智能客服  
> 学习目标：不是系统学习全部数据库理论，而是掌握**能直接用于当前智能客服项目**的数据库知识，并最终能够把商品、会话、消息、订单、人工接管、状态记录等业务数据稳定地存储和管理起来。

---

# 1. 为什么现在需要学习数据库

当前智能客服已经不再只是一个“问知识库问题”的 RAG Demo。

项目已经开始涉及：

- 商品信息；
- 闲鱼商品 ID 映射；
- 买家消息；
- 会话状态；
- AI 自动回复；
- 人工接管；
- 订单查询；
- MCP；
- RAG 知识库；
- 商品事实；
- 后续可能出现的前端客服工作台；
- 后续可能出现的多用户、多商品、多订单、多会话。

在项目早期，使用 `items.json`、Markdown 文件、内存变量来保存数据没有问题。

例如：

```json
{
  "item_id": "CANON_FTB_001",
  "title": "Canon FTb 胶片相机",
  "listed_price_cents": 150000,
  "sale_status": "listed"
}
```

这种方式适合：

- 数据少；
- 单人开发；
- 快速验证；
- 不需要复杂查询；
- 不需要多人同时读写；
- 不需要完整历史记录。

但是一旦项目继续向真实客服系统发展，就会出现这些问题：

```text
同一个商品会不断修改
同一个买家会产生大量消息
同一个会话会发生多次状态变化
订单状态会更新
人工客服会接管
系统需要记录历史
程序可能重复收到同一条消息
多个请求可能同时处理同一个会话
需要快速查询最近 20 条消息
需要查询某个商品关联的全部会话
需要保证订单状态不能被写乱
```

这时，仅靠 JSON 文件就不够了。

因此，后续数据库学习的核心目标不是：

> “学会数据库这门课程。”

而是：

> **让智能客服拥有可靠、可查询、可维护、可扩展的业务数据基础。**

---

# 2. 先建立整个项目的数据架构认知

这个智能客服以后最好不要让一种数据库承担所有任务。

更合理的结构是：

```text
PostgreSQL
    ↓
保存业务事实和长期业务数据

Qdrant
    ↓
保存向量和知识检索索引

Redis
    ↓
保存缓存、锁、短期状态

LLM
    ↓
理解问题、组合上下文、生成回答
```

最重要的是理解这四者的职责完全不同。

---

## 2.1 PostgreSQL：业务事实数据库

PostgreSQL 主要保存：

```text
商品
订单
买家
会话
消息
人工接管状态
状态变化历史
渠道商品映射
系统事件
客服操作记录
```

例如：

```text
CANON_FTB_001 当前售价是多少？
这个商品有没有卖掉？
这个订单现在是什么状态？
这个会话现在是 AI 还是人工？
买家刚才说了什么？
```

这些都属于**业务事实**。

业务事实必须有一个可靠的数据来源。

---

## 2.2 Qdrant：知识检索数据库

Qdrant 负责的是：

```text
Markdown
    ↓
Chunk
    ↓
Embedding
    ↓
Vector
    ↓
向量检索
```

适合存储和检索：

- 物流规则；
- 售后政策；
- 退款规则；
- 付款规则；
- 平台说明；
- 商品长文本知识；
- FAQ；
- 其他需要语义搜索的内容。

例如买家问：

```text
普通商品一般多久发货？
```

适合走：

```text
RAG
→ Qdrant
→ shipping.md
→ 发货规则
```

但是买家问：

```text
我这笔订单发货了吗？
```

不能让 Qdrant 回答。

应该查询订单系统或 PostgreSQL。

---

## 2.3 Redis：临时数据和高频状态

Redis 不应该作为最终业务事实来源。

它适合：

- 缓存；
- 分布式锁；
- 限流；
- 临时会话状态；
- 幂等辅助；
- 短期任务状态；
- 过期数据。

例如：

```text
chat_lock:CHAT001 = locked
```

表示某个聊天正在处理。

或者：

```text
rate_limit:user_001 = 7
```

表示某个用户一分钟已经请求 7 次。

Redis 最大的特点之一是：

> 数据可以设置过期时间。

因此它非常适合短期数据。

---

## 2.4 LLM 不是数据库

这是后续整个项目必须坚持的原则。

LLM 可以：

- 理解用户在问什么；
- 判断问题类型；
- 读取数据库提供的事实；
- 读取 RAG 提供的知识；
- 将事实组织成自然语言；
- 决定是否澄清；
- 决定是否转人工。

但是 LLM 不应该：

- 自己猜库存；
- 自己猜订单状态；
- 自己猜商品有没有维修过；
- 自己猜是否摔过；
- 自己把未知事实当成“没有”。

例如数据库中只有：

```text
condition = 正常使用痕迹
```

用户问：

```text
这台相机有没有摔过？
```

不能推断：

```text
没有摔过。
```

因为：

```text
正常使用痕迹
≠
明确没有摔过
```

正确处理应该是：

```text
没有足够事实
→ 无法确定
→ 澄清 / 转人工
```

---

# 3. 第一阶段：必须掌握的关系型数据库基础

这一部分是数据库学习的真正起点。

需要掌握：

1. Database
2. Table
3. Row
4. Column
5. Primary Key
6. Foreign Key
7. One-to-Many
8. Many-to-Many
9. NULL
10. 数据类型

这些概念后面所有数据库设计都会用到。

---

# 4. Database、Table、Row、Column

可以先把 PostgreSQL 理解成一个非常强大的结构化数据管理系统。

一个数据库中可以有很多表：

```text
customer_service
│
├── items
├── buyers
├── conversations
├── messages
├── orders
├── order_items
└── handoff_events
```

---

## 4.1 Table：表

例如：

```text
items
```

专门保存商品。

可能长这样：

| item_id | title | price_cents | sale_status |
|---|---|---:|---|
| CANON_FTB_001 | Canon FTb 胶片相机 | 150000 | listed |
| ITEM_002 | 咖啡机 | 98000 | sold |

---

## 4.2 Row：行

一行表示一条记录。

例如：

```text
CANON_FTB_001 | Canon FTb 胶片相机 | 150000 | listed
```

表示一个具体商品。

---

## 4.3 Column：列

列表示一种字段。

例如：

```text
item_id
title
price_cents
sale_status
```

每一个字段应该有相对明确的意义。

---

# 5. 数据类型

设计数据库时，不只是写一个字段名字，还要决定它是什么类型。

常见类型：

```text
TEXT
VARCHAR
INTEGER
BIGINT
BOOLEAN
TIMESTAMP
JSONB
NUMERIC
```

例如：

```sql
title TEXT
price_cents INTEGER
is_active BOOLEAN
created_at TIMESTAMP
raw_payload JSONB
```

---

## 5.1 金额不要优先使用 float

例如商品价格：

```text
1500.00 元
```

不要简单存：

```text
1500.00 float
```

更简单可靠的方式是：

```text
150000 cents
```

也就是：

```text
price_cents = 150000
```

单位是分。

这样可以避免很多浮点数精度问题。

---

# 6. Primary Key：主键

主键是关系型数据库最重要的基础概念之一。

主键代表：

> 一条记录的唯一身份。

例如商品：

```text
item_id = CANON_FTB_001
```

这个 ID 应该唯一。

数据库可以定义：

```sql
CREATE TABLE items (
    item_id TEXT PRIMARY KEY,
    title TEXT NOT NULL
);
```

这样：

```text
CANON_FTB_001
```

不能重复出现两次。

---

## 6.1 为什么主键重要

假设没有主键：

```text
Canon FTb
Canon FTb
Canon FTb
```

你无法确定到底是哪一个商品。

有主键以后：

```text
CANON_FTB_001
CANON_FTB_002
CANON_FTB_003
```

即使标题一样，也可以准确区分。

---

# 7. Foreign Key：外键

外键用于建立不同表之间的关系。

例如：

```text
items
```

保存商品。

```text
messages
```

保存聊天消息。

消息表中可以有：

```text
item_id = CANON_FTB_001
```

于是：

```text
messages.item_id
        ↓
items.item_id
```

表示：

> 这条消息属于这个商品。

数据库可以使用：

```sql
FOREIGN KEY (item_id)
REFERENCES items(item_id)
```

这样数据库可以帮助保证：

```text
messages 里面引用的商品
必须是真实存在的商品
```

---

# 8. 一对多关系

智能客服里最常见的关系就是一对多。

例如：

```text
一个会话
    ↓
很多消息
```

就是：

```text
Conversation 1
    ├── Message 1
    ├── Message 2
    ├── Message 3
    └── Message 4
```

对应数据库：

```text
conversations
conversation_id = CHAT001
```

消息表：

```text
messages

MSG001 → CHAT001
MSG002 → CHAT001
MSG003 → CHAT001
```

这就是：

```text
One To Many
一对多
```

---

# 9. 多对多关系

后面订单系统可能会出现多对多。

例如：

```text
一个订单
可以有多个商品

一个商品
也可能出现在多个订单中
```

不能简单只在订单表放一个 `item_id`。

更合理的是：

```text
orders
```

和：

```text
items
```

中间加：

```text
order_items
```

例如：

```text
order_items

order_id    item_id
ORDER001    ITEM001
ORDER001    ITEM002
ORDER002    ITEM001
```

这叫关联表。

---

# 10. NULL：必须真正理解

`NULL` 非常重要。

它不是：

```text
false
```

也不是：

```text
0
```

也不是：

```text
空字符串
```

它表示：

> 未知 / 没有值。

例如：

```text
has_been_repaired = false
```

表示：

> 已经确认：没有维修过。

而：

```text
has_been_repaired = NULL
```

表示：

> 不知道有没有维修过。

对 AI 客服来说，这个差别非常重要。

---

## 10.1 AI 客服里的核心规则

以后数据库应该明确区分：

```text
YES
NO
UNKNOWN
```

而不是只有：

```text
YES
NO
```

因为真实世界中有大量信息是不确定的。

例如：

```text
有没有进水？
有没有摔过？
有没有拆机？
有没有维修？
电池健康度多少？
原包装还在不在？
```

如果卖家没有提供信息：

```text
UNKNOWN
```

AI 应该：

```text
不能编造
→ 澄清
→ 转人工
```

---

# 11. 第二阶段：必须掌握的 SQL

前期不需要掌握所有 SQL。

先掌握：

```text
SELECT
INSERT
UPDATE
DELETE
WHERE
ORDER BY
LIMIT
JOIN
GROUP BY
COUNT
```

这些已经能够完成智能客服绝大部分基础数据操作。

---

# 12. SELECT：查询

最基础：

```sql
SELECT *
FROM items;
```

表示：

> 查询全部商品。

---

## 12.1 根据商品 ID 查询

```sql
SELECT *
FROM items
WHERE item_id = 'CANON_FTB_001';
```

这就是：

```text
从 items
找到 item_id 等于 CANON_FTB_001 的商品
```

---

## 12.2 只查需要的字段

不要每次都：

```sql
SELECT *
```

可以：

```sql
SELECT
    item_id,
    title,
    price_cents,
    sale_status
FROM items
WHERE item_id = 'CANON_FTB_001';
```

---

# 13. INSERT：新增

新增商品：

```sql
INSERT INTO items (
    item_id,
    title,
    price_cents,
    sale_status
)
VALUES (
    'CANON_FTB_001',
    'Canon FTb 胶片相机',
    150000,
    'listed'
);
```

---

# 14. UPDATE：修改

商品卖出以后：

```sql
UPDATE items
SET sale_status = 'sold'
WHERE item_id = 'CANON_FTB_001';
```

最重要的是：

> UPDATE 一般必须认真检查 WHERE。

如果忘记：

```sql
WHERE item_id = ...
```

可能会把整张表全部修改。

---

# 15. DELETE：删除

例如：

```sql
DELETE FROM items
WHERE item_id = 'CANON_FTB_001';
```

真实系统中，很多业务数据不会直接物理删除。

可能会采用：

```text
deleted_at
is_deleted
```

这种软删除方式。

但项目早期可以先理解普通 DELETE。

---

# 16. CRUD

以后经常会看到：

```text
CRUD
```

分别是：

```text
Create
Read
Update
Delete
```

对应：

```text
INSERT
SELECT
UPDATE
DELETE
```

一个基本业务模块通常都会涉及 CRUD。

---

# 17. WHERE：过滤条件

例如：

```sql
SELECT *
FROM items
WHERE sale_status = 'listed';
```

表示：

> 只查询还在售的商品。

多个条件：

```sql
SELECT *
FROM items
WHERE sale_status = 'listed'
  AND price_cents < 200000;
```

---

# 18. ORDER BY：排序

例如查询最新消息：

```sql
SELECT *
FROM messages
WHERE conversation_id = 'CHAT001'
ORDER BY created_at DESC;
```

表示：

> 按创建时间，从最新到最旧排列。

---

# 19. LIMIT：限制结果数量

客服系统非常常见：

```sql
SELECT *
FROM messages
WHERE conversation_id = 'CHAT001'
ORDER BY created_at DESC
LIMIT 20;
```

表示：

> 取最近 20 条消息。

这以后可以直接用于：

```text
会话上下文
最近 N 轮消息
客服工作台
```

---

# 20. JOIN：必须掌握

JOIN 是后续最重要的 SQL 技能之一。

例如：

```text
messages
```

只保存：

```text
message_id
conversation_id
item_id
content
```

而商品名称在：

```text
items
```

中。

要一起查询：

```sql
SELECT
    messages.content,
    items.title
FROM messages
JOIN items
    ON messages.item_id = items.item_id;
```

这样就把：

```text
messages
+
items
```

关联起来。

---

## 20.1 智能客服中常见 JOIN

后面会大量出现：

```text
conversation JOIN messages
item JOIN item_facts
order JOIN order_items
order_items JOIN items
conversation JOIN buyer
conversation JOIN handoff_events
```

所以 JOIN 必须真正理解。

---

# 21. 第三阶段：数据库表设计

这个阶段不再是“学 SQL”。

而是开始学习：

> 一个真实业务应该怎么拆表。

数据库设计的核心不是：

> 表越多越专业。

而是：

> 数据职责清楚、关系清楚、重复少、事实来源明确。

---

# 22. 当前智能客服推荐的核心表

后续可以逐步演进到：

```text
items
item_facts
channel_item_mappings

buyers

conversations
messages

orders
order_items

handoff_events
```

不是要求现在一次性全部实现。

而是把它作为目标结构。

---

# 23. items：商品主表

建议负责保存：

```text
商品身份
商品标题
价格
销售状态
商品描述
数据来源
创建时间
更新时间
```

示例：

```text
items

item_id
title
listed_price_cents
sale_status
listing_description
data_source
created_at
updated_at
```

---

# 24. listing_description：商品原始描述

你之前已经意识到：

> 如果 items.json 里只有商品标题和价格，AI 根本不知道卖家商品详情页写了什么。

这是正确的。

因此商品表可以保存：

```text
listing_description
```

例如：

```text
Canon FTb 胶片相机。
正常使用痕迹。
快门正常。
测光正常。
镜头轻微灰尘。
无原包装。
带皮套。
```

这样系统至少拥有商品详情页的原始事实来源。

---

# 25. item_facts：结构化商品事实

只有长文本描述还不够。

因为程序很难稳定判断：

```text
有没有原包装？
有没有维修过？
快门正常吗？
```

所以后续可以增加：

```text
item_facts
```

例如：

| item_id | fact_key | fact_value |
|---|---|---|
| CANON_FTB_001 | shutter | 正常 |
| CANON_FTB_001 | light_meter | 正常 |
| CANON_FTB_001 | original_box | 无 |
| CANON_FTB_001 | repaired | unknown |
| CANON_FTB_001 | condition | 正常使用痕迹 |

这会让客服回答更加稳定。

---

# 26. 商品事实设计的重要原则

商品数据应该尽量区分：

```text
卖家明确提供的事实
系统自己计算的数据
AI 推断的数据
未知信息
```

不要混在一起。

例如：

```text
repaired = no
```

必须意味着：

> 卖家明确确认没有维修过。

而不是：

> AI 从描述里没有看到“维修”两个字，所以认为没有维修。

---

# 27. channel_item_mappings：渠道商品映射

你当前项目已经有：

```text
闲鱼商品 ID
        ↓
内部 item_id
```

后续非常适合单独做一张映射表。

例如：

```text
channel_item_mappings

channel
external_item_id
item_id
```

数据：

```text
xianyu
1084130180117
CANON_FTB_001
```

这样内部业务不需要到处依赖闲鱼原始 ID。

---

# 28. 为什么需要内部 item_id

因为以后可能不仅有闲鱼。

可能出现：

```text
闲鱼
淘宝
微信
独立站
线下
```

同一个商品可以有不同平台 ID。

内部统一使用：

```text
CANON_FTB_001
```

平台只是映射：

```text
xianyu:1084130180117
        ↓
CANON_FTB_001
```

这样架构更稳定。

---

# 29. conversations：会话表

保存一段买家会话的整体状态。

例如：

```text
conversation_id
buyer_id
item_id
channel
state
created_at
updated_at
```

其中：

```text
state
```

可能是：

```text
AUTO
HUMAN
PAUSED
CLOSED
```

---

# 30. messages：消息表

每条消息单独保存。

例如：

```text
message_id
conversation_id
sender_type
content
message_type
external_message_id
created_at
raw_payload
```

`sender_type` 可能是：

```text
BUYER
AI
SELLER
SYSTEM
```

这样可以完整还原聊天历史。

---

# 31. 为什么 conversation 和 message 要分开

因为：

```text
conversation
```

是一段聊天。

而：

```text
message
```

是一条消息。

关系：

```text
一个 conversation
    ↓
很多 message
```

不能把所有消息直接塞成一个超大 JSON。

否则：

- 查询最近消息麻烦；
- 统计麻烦；
- 更新麻烦；
- 搜索麻烦；
- 并发写入麻烦；
- 单条消息去重麻烦。

---

# 32. orders：订单表

如果以后订单数据真正进入 PostgreSQL，可以设计：

```text
orders

order_id
buyer_id
status
total_amount_cents
created_at
paid_at
shipped_at
completed_at
```

常见状态：

```text
pending_payment
paid
shipped
completed
cancelled
refunding
refunded
```

具体状态要由真实业务决定。

---

# 33. order_items：订单商品表

如果订单可以有多个商品：

```text
order_items

order_id
item_id
quantity
unit_price_cents
```

这样：

```text
orders
    ↓
order_items
    ↓
items
```

就能完整关联。

---

# 34. handoff_events：人工接管事件

当前项目已经有：

```text
AI 自动回复
人工接管
暂停
恢复
```

不能只保存：

```text
conversation.state = HUMAN
```

还应该考虑保存历史：

```text
handoff_events
```

例如：

```text
event_id
conversation_id
from_state
to_state
reason
operator
created_at
```

历史：

```text
19:00 AUTO → HUMAN
19:15 HUMAN → AUTO
19:40 AUTO → PAUSED
```

这样以后可以知道：

- 谁接管的；
- 什么时候接管的；
- 为什么接管；
- 接管了多少次；
- AI 自动处理了多久。

---

# 35. 当前状态 + 历史事件

这是一个非常实用的设计思想。

```text
conversations
↓
保存当前状态

handoff_events
↓
保存状态变化历史
```

也就是：

```text
Current State
+
Event History
```

这样既能快速查询：

```text
现在是谁在处理？
```

也可以查询：

```text
过去发生过什么？
```

---

# 36. 第四阶段：约束 Constraint

数据库不是单纯“保存数据”。

它还应该防止错误数据进入系统。

常见约束：

```text
PRIMARY KEY
FOREIGN KEY
UNIQUE
NOT NULL
CHECK
```

---

# 37. NOT NULL

例如：

```text
title
```

商品标题必须存在。

可以：

```sql
title TEXT NOT NULL
```

如果程序准备写入：

```text
title = NULL
```

数据库直接拒绝。

---

# 38. UNIQUE

例如闲鱼消息：

```text
external_message_id
```

不应该重复。

可以：

```sql
UNIQUE(external_message_id)
```

这样网络重试导致同一条消息再次进入时，数据库可以帮助发现重复。

---

# 39. CHECK

例如：

```text
price_cents
```

不能小于 0。

```sql
CHECK (price_cents >= 0)
```

数据库就会拒绝：

```text
price_cents = -1000
```

---

# 40. 数据库是最后一道数据防线

应用代码可能有 Bug。

AI 可能输出错误。

网络可能重试。

多个请求可能同时执行。

因此不能只依赖 Python：

```text
if price >= 0:
```

数据库本身也应该保护关键规则。

---

# 41. 第五阶段：索引 Index

索引的作用可以理解为：

> 给数据库建立目录。

没有索引：

```text
一行一行扫描
```

有索引：

```text
通过目录快速定位
```

---

# 42. 哪些字段可能需要索引

这个项目后面常见的索引字段：

```text
item_id
buyer_id
conversation_id
order_id
external_message_id
created_at
sale_status
```

例如：

```sql
CREATE INDEX idx_messages_conversation_id
ON messages(conversation_id);
```

这样查某个会话的消息会更快。

---

# 43. 组合索引

以后还会遇到：

```text
conversation_id + created_at
```

因为经常执行：

```sql
SELECT *
FROM messages
WHERE conversation_id = 'CHAT001'
ORDER BY created_at DESC
LIMIT 20;
```

这时组合索引可能更有效。

但前期不需要过度优化。

原则：

> 先根据真实查询需求加索引，不要见字段就加。

---

# 44. 为什么索引不是越多越好

因为每次写入数据：

```text
INSERT
UPDATE
DELETE
```

数据库还要同步维护索引。

索引过多会：

- 占空间；
- 写入变慢；
- 维护复杂。

所以索引应该围绕真实查询场景设计。

---

# 45. 第六阶段：事务 Transaction

事务是后续真正进入业务系统后必须掌握的内容。

核心思想：

> 一组操作要么全部成功，要么全部失败。

---

# 46. 为什么订单需要事务

假设库存只有：

```text
1
```

用户购买：

```text
步骤1：库存 -1
步骤2：创建订单
步骤3：记录付款
```

执行到步骤 2 时程序崩了。

如果没有事务：

```text
库存 = 0
订单不存在
```

数据已经不一致。

---

# 47. Transaction

数据库可以：

```text
BEGIN

库存 -1
创建订单
记录付款

成功
↓
COMMIT

任何一步失败
↓
ROLLBACK
```

这样就不会出现一半成功、一半失败。

---

# 48. ACID 需要知道，但不用死背

事务通常会提：

```text
Atomicity
Consistency
Isolation
Durability
```

当前阶段理解含义即可：

```text
Atomicity
一组操作整体成功或整体失败

Consistency
数据前后保持合法

Isolation
并发事务之间尽量不互相破坏

Durability
提交后的数据应该可靠保存
```

不需要现在研究所有理论细节。

---

# 49. 第七阶段：并发

智能客服天然是并发系统。

因为可能同时出现：

```text
多个买家
多个会话
同一个买家连续发消息
卖家人工接管
AI 正在生成回答
系统同时更新状态
```

---

# 50. 一个真实并发问题

假设：

```text
19:00:00
买家：在吗？

19:00:00.1
买家：最低多少钱？
```

系统可能同时启动：

```text
Worker A
Worker B
```

两边都读取：

```text
state = AUTO
```

然后都调用 LLM。

这时卖家：

```text
切换为 HUMAN
```

如果系统没有并发控制：

```text
AI A 发了一条
AI B 又发了一条
人工客服也发了一条
```

这就是实际工程问题。

---

# 51. 需要逐步理解的并发工具

以后要学习：

```text
数据库事务
行锁
乐观锁
version 字段
Redis Lock
状态二次校验
```

当前阶段不用一次学完。

最重要的是先理解：

> 读取状态以后，到真正发送消息之前，状态可能已经发生变化。

因此不能简单：

```text
读取 AUTO
↓
生成 10 秒
↓
直接发送
```

应该再次检查：

```text
发送前再确认当前状态
```

你现在项目中“卖家接管后旧 AI 答案不能发送”，本质上已经是在处理这个问题。

---

# 52. 第八阶段：幂等 Idempotency

这是闲鱼消息链路非常重要的知识。

网络系统经常会重试。

例如闲鱼发送：

```text
message_id = XY12345
```

你的服务器没有及时确认。

平台可能再次发送：

```text
XY12345
```

实际上：

```text
同一条消息
```

发送了两次。

---

# 53. 没有幂等会发生什么

第一次：

```text
收到 XY12345
↓
AI 回复
```

第二次：

```text
再次收到 XY12345
↓
AI 又回复一次
```

买家看到：

```text
好的，可以的。

好的，可以的。
```

这是严重体验问题。

---

# 54. 数据库如何帮助幂等

保存：

```text
external_message_id
```

并设置：

```sql
UNIQUE(external_message_id)
```

第二次收到：

```text
XY12345
```

数据库发现已经存在：

```text
已处理
↓
忽略
```

这是最常见的幂等设计之一。

---

# 55. 幂等以后还会用于

不仅消息系统。

还包括：

```text
支付回调
订单创建
退款
Webhook
任务重试
外部 API 回调
```

所以一定要真正理解这个概念。

---

# 56. 第九阶段：Source of Truth

这是 AI 系统最重要的数据设计原则之一。

Source of Truth：

> 某一种数据最终到底以谁为准。

---

# 57. 例子：商品状态冲突

PostgreSQL：

```text
CANON_FTB_001
status = sold
```

Qdrant 中旧文本：

```text
CANON_FTB_001
status = listed
```

到底相信谁？

必须提前规定：

```text
PostgreSQL
=
商品状态 Source of Truth
```

因此系统必须认为：

```text
sold
```

才是真实状态。

---

# 58. 建议明确的数据责任

可以建立这样的规则：

```text
商品价格、库存、状态
→ PostgreSQL

订单状态
→ PostgreSQL / 实际订单系统

客服规则
→ Markdown + Qdrant

RAG 向量
→ Qdrant

临时锁和缓存
→ Redis

回答
→ LLM
```

不要让多个系统同时成为同一个事实的“最终真相”。

---

# 59. 第十阶段：JSONB

PostgreSQL 支持：

```text
JSONB
```

可以直接保存 JSON。

这个功能对闲鱼消息非常实用。

---

# 60. 什么适合正常字段

经常查询、排序、约束的数据：

```text
message_id
conversation_id
item_id
created_at
status
price
```

应该优先使用普通列。

---

# 61. 什么适合 JSONB

不稳定、渠道特有、字段很多、后续可能改变的原始内容：

```text
raw_payload
platform_metadata
extra_data
```

可以使用 JSONB。

例如：

```text
messages

message_id
conversation_id
content
created_at
raw_payload JSONB
```

其中：

```text
raw_payload
```

可以完整保存闲鱼的原始事件。

这样以后发现解析逻辑有问题，还能查看最初收到的原始数据。

---

# 62. 不要把所有东西都塞 JSONB

错误设计：

```text
items

id
data JSONB
```

然后：

```json
{
  "title": "...",
  "price": "...",
  "status": "...",
  "description": "...",
  "created_at": "..."
}
```

这样相当于重新把 PostgreSQL 当成 JSON 文件。

原则：

> 稳定、关键、经常查询的数据用列。  
> 变化大、附加型数据再用 JSONB。

---

# 63. 第十一阶段：时间字段

真实系统里时间非常重要。

常见字段：

```text
created_at
updated_at
deleted_at
paid_at
shipped_at
closed_at
last_message_at
```

以后调试时经常会问：

```text
这条消息什么时候进来的？
什么时候 AI 开始处理？
什么时候转人工？
订单什么时候发货？
```

没有时间字段，后续很难调查问题。

---

# 64. 推荐基本规则

大部分业务表至少考虑：

```text
created_at
updated_at
```

事件表一般至少：

```text
created_at
```

时间最好统一时区处理策略。

例如数据库保存 UTC，展示层转换成本地时间。

但项目早期首先保证：

> 时间字段有统一规则，不要一部分有时区、一部分没有时区。

---

# 65. 第十二阶段：Repository

你当前项目里非常值得引入 Repository 思想。

现在如果业务代码到处：

```python
open("items.json")
```

以后迁移 PostgreSQL 时会很痛苦。

---

# 66. Repository 的目的

让业务层只知道：

```python
item_repository.get_by_id(item_id)
```

而不知道底层到底是：

```text
JSON
PostgreSQL
API
其他服务
```

---

# 67. 当前阶段可以先抽象

例如：

```text
ItemRepository
```

提供：

```text
get_by_id
list_items
create_item
update_item
```

现在实现：

```text
JsonItemRepository
```

以后再实现：

```text
PostgresItemRepository
```

上层：

```text
ChatService
```

几乎不用改。

---

# 68. 推荐分层

以后项目可以逐渐靠近：

```text
API
↓
Service
↓
Repository
↓
Database
```

例如：

```text
POST /chat
↓
ChatService
↓
ItemRepository
↓
PostgreSQL
```

这样比在 Service 里面直接写 SQL 更容易维护。

---

# 69. 第十三阶段：SQLAlchemy

你的项目是 Python + FastAPI。

后面最适合学习：

```text
SQLAlchemy
```

它负责：

> 让 Python 代码操作关系型数据库。

---

# 70. ORM 的基本概念

ORM：

```text
Object Relational Mapping
```

把：

```text
Python 类
```

映射到：

```text
数据库表
```

例如：

```python
class Item:
    item_id
    title
    price_cents
```

对应：

```text
items
```

表。

这样业务代码不需要所有地方都手写 SQL。

---

# 71. 不要因为 ORM 就完全不学 SQL

这是非常重要的。

即使使用 SQLAlchemy，也必须懂：

```text
SELECT
JOIN
WHERE
INDEX
TRANSACTION
```

因为：

- ORM 最终还是生成 SQL；
- 性能问题需要看 SQL；
- 查询逻辑需要理解；
- 数据库错误需要排查；
- 复杂查询很难完全绕开 SQL 思维。

所以正确学习顺序：

```text
先懂 SQL
↓
再用 SQLAlchemy
```

---

# 72. 第十四阶段：Alembic

数据库结构会不断变化。

今天：

```text
items

item_id
title
price
```

明天增加：

```text
listing_description
```

后天增加：

```text
condition
```

不能靠手工随便改数据库。

---

# 73. Migration

Migration：

> 数据库结构版本迁移。

例如：

```text
001_create_items
002_add_listing_description
003_create_messages
004_add_external_message_id
```

这和 Git 很像。

```text
Git
↓
代码版本

Alembic
↓
数据库结构版本
```

---

# 74. 为什么 Migration 必须学

如果只有自己电脑：

```text
手动改数据库
```

似乎没问题。

以后：

```text
开发环境
测试环境
服务器
同事电脑
```

都需要一样的表结构。

Migration 可以保证：

```text
大家按照同一套步骤升级数据库
```

---

# 75. 第十五阶段：Redis

Redis 放在 PostgreSQL 之后学习。

不要反过来。

因为当前项目最需要的是：

```text
可靠业务数据
```

而不是先增加缓存复杂度。

---

# 76. Redis 重点学什么

只需要围绕项目学习：

```text
Key-Value
TTL
Cache
Lock
Rate Limit
简单计数
临时 Session
```

---

# 77. TTL

TTL：

> Time To Live

表示：

```text
这个数据多久以后自动过期。
```

例如：

```text
chat_lock:CHAT001
```

设置：

```text
30 秒后自动消失
```

即使程序异常退出，锁也不会永远存在。

---

# 78. Cache

假设某个商品信息被频繁读取。

可以：

```text
第一次
↓
PostgreSQL
↓
保存 Redis

后续
↓
优先 Redis
```

但必须理解：

> Redis 是缓存，PostgreSQL 才是事实来源。

---

# 79. Lock

Redis 可以帮助实现：

```text
CHAT001
同一时间只允许一个处理任务
```

逻辑：

```text
收到消息
↓
尝试获得 chat_lock:CHAT001
↓
成功
    处理
失败
    等待 / 跳过 / 排队
```

真实实现还需要考虑锁超时、释放、异常等问题。

当前阶段先理解用途即可。

---

# 80. Rate Limit

例如限制：

```text
同一个买家
1 分钟最多触发 20 次 AI 调用
```

Redis 很适合做这种短期计数。

---

# 81. 第十六阶段：Qdrant 与 PostgreSQL 的边界

这是当前项目必须特别清楚的内容。

---

# 82. Qdrant 适合回答

```text
平台退款规则是什么？
普通订单多久发货？
退货条件是什么？
如何联系客服？
商品的一段长文本介绍中有哪些相关内容？
```

这些属于：

```text
知识检索
```

---

# 83. PostgreSQL 适合回答

```text
这个商品现在多少钱？
这个商品卖掉了吗？
这条消息处理过了吗？
当前会话是谁在处理？
这个订单已经发货了吗？
买家上一条消息是什么？
```

这些属于：

```text
业务事实
```

---

# 84. MCP 的位置

MCP 本身不是数据库。

MCP 更像：

> AI / 应用访问外部工具和业务系统的一种接口方式。

例如：

```text
LLM
↓
MCP Tool
↓
订单系统
↓
订单数据库
```

因此：

```text
数据库
```

负责存数据。

```text
MCP
```

负责提供访问能力。

两者职责不同。

---

# 85. 第十七阶段：聊天历史和 Memory

以后如果要实现多轮对话：

```text
messages
```

就是最可靠的历史来源之一。

例如需要最近 3 轮：

```sql
SELECT *
FROM messages
WHERE conversation_id = 'CHAT001'
ORDER BY created_at DESC
LIMIT 6;
```

因为：

```text
1轮
≈
用户消息 + AI消息
```

三轮通常是 6 条左右。

实际实现可以根据 sender_type 进一步处理。

---

# 86. 长期聊天记录不要只放 Redis

Redis 可以缓存最近几轮。

但完整历史应该更适合：

```text
PostgreSQL
```

原因：

- Redis 可能过期；
- Redis 主要服务性能；
- 聊天记录需要长期追踪；
- 后续需要审计；
- 前端需要展示历史会话。

推荐：

```text
PostgreSQL
↓
长期历史

Redis
↓
最近上下文缓存
```

---

# 87. 第十八阶段：审计和可追踪性

AI 客服上线以后，一个非常重要的问题是：

```text
为什么它当时这样回复？
```

如果什么都没记录，就无法排查。

---

# 88. 后续可以记录的重要信息

例如：

```text
message_id
conversation_id
item_id
route
action
AI / HUMAN
retrieved_sources
model
response
created_at
```

甚至后续可以单独建立：

```text
ai_response_logs
```

不过项目早期不要过度设计。

先保证：

```text
输入
输出
会话
状态变化
时间
```

能追踪。

---

# 89. 第十九阶段：数据一致性

后续必须逐渐建立一个思想：

> 同一个事实不要在多个地方被独立维护。

例如：

```text
items.price
```

在 PostgreSQL 有一份。

如果又在：

```text
Qdrant
JSON
Redis
Prompt
```

分别手工维护价格，就会出现冲突。

---

# 90. 更合理的方式

```text
PostgreSQL
↓
唯一价格事实
```

其他系统：

```text
Redis
↓
缓存

Qdrant
↓
只用于知识检索

Prompt
↓
运行时从数据库注入
```

避免多个来源各自修改。

---

# 91. 第二十阶段：备份

当数据库里开始出现真实业务数据后，需要理解：

```text
Backup
Restore
```

即：

```text
备份
恢复
```

至少知道：

> 数据库坏了以后，能不能恢复？

项目学习阶段不需要立即建立复杂灾备。

但是一旦保存真实买家、订单、商品信息，就必须逐渐考虑备份。

---

# 92. 第二十一阶段：数据库安全

数据库连接信息不要硬编码。

错误：

```python
password = "123456"
```

正确方向：

```text
.env
环境变量
Secret 管理
```

至少包括：

```text
DATABASE_URL
DB_HOST
DB_PORT
DB_USER
DB_PASSWORD
DB_NAME
```

---

# 93. 最小权限原则

后面如果正式部署：

```text
应用数据库账号
```

不应该拥有所有管理员权限。

基本思想：

> 应用只拥有它真正需要的权限。

当前本地开发阶段先知道概念即可。

---

# 94. 第二十二阶段：数据库测试

数据库功能不能只看：

```text
程序没报错
```

要设计测试。

---

# 95. 基础测试

例如：

```text
新增商品
↓
能查询到

修改状态
↓
查询结果变为 sold

重复 external_message_id
↓
数据库拒绝

不存在 item_id
↓
不能创建错误外键消息
```

---

# 96. 事务测试

例如：

```text
步骤2故意报错
↓
步骤1修改必须回滚
```

确保事务真正生效。

---

# 97. 幂等测试

输入两次：

```text
external_message_id = XY123
```

最终：

```text
只保存一条
只触发一次业务处理
```

---

# 98. 并发测试

后期可以测试：

```text
同一会话
两个请求同时进入
```

最终不应该导致：

- 状态混乱；
- 重复发送；
- 人工接管后 AI 继续回复。

---

# 99. 当前项目数据库的推荐最终结构

可以把未来目标想象成：

```text
                        ┌──────────────┐
                        │   闲鱼买家    │
                        └──────┬───────┘
                               ↓
                        ┌──────────────┐
                        │ 消息接收层    │
                        └──────┬───────┘
                               ↓
                        ┌──────────────┐
                        │ ChatService  │
                        └──────┬───────┘
                               ↓
                      Intent / Route
                      /      |       \
                     /       |        \
                    ↓        ↓         ↓

             商品事实      订单事实      规则知识
                ↓             ↓           ↓
          PostgreSQL         MCP        Qdrant
                \             |          /
                 \            |         /
                  └──── Context ───────┘
                           ↓
                          LLM
                           ↓
                         回复

与此同时：

PostgreSQL
├── items
├── item_facts
├── channel_item_mappings
├── buyers
├── conversations
├── messages
├── orders
├── order_items
└── handoff_events

Redis
├── chat_lock
├── cache
├── rate_limit
└── temporary_state
```

---

# 100. 推荐学习顺序

不要同时学所有东西。

按照下面顺序最适合当前项目。

---

## 阶段 1：关系型数据库基本概念

必须掌握：

```text
Database
Table
Row
Column
Primary Key
Foreign Key
One-to-Many
Many-to-Many
NULL
Data Type
```

完成标准：

> 能看懂一张数据库表，并能解释不同表为什么要关联。

---

## 阶段 2：基础 SQL

学习：

```text
SELECT
INSERT
UPDATE
DELETE
WHERE
ORDER BY
LIMIT
JOIN
COUNT
GROUP BY
```

完成标准：

> 能独立写 SQL 查询商品、会话、消息和订单。

---

## 阶段 3：直接设计自己的智能客服数据库

不要使用：

```text
学生管理系统
图书管理系统
```

来练习。

直接设计：

```text
items
conversations
messages
```

第一版只做这三张表即可。

完成标准：

> 能把当前 items.json 和聊天记录映射成数据库结构。

---

## 阶段 4：约束与索引

学习：

```text
PRIMARY KEY
FOREIGN KEY
UNIQUE
NOT NULL
CHECK
INDEX
```

完成标准：

> 数据库能够主动拒绝重复消息、无效价格、错误关联等数据。

---

## 阶段 5：Repository

先让业务代码不再依赖：

```text
open("items.json")
```

而是依赖：

```text
ItemRepository
```

完成标准：

> 上层 ChatService 不关心底层是 JSON 还是 PostgreSQL。

---

## 阶段 6：PostgreSQL + SQLAlchemy

学习：

```text
连接 PostgreSQL
SQLAlchemy Model
Session
基础 CRUD
关系映射
```

完成标准：

> items 可以真正由 PostgreSQL 提供。

---

## 阶段 7：Alembic

学习：

```text
migration
upgrade
downgrade
revision
```

完成标准：

> 增加数据库字段时，不再手工修改数据库。

---

## 阶段 8：事务

学习：

```text
BEGIN
COMMIT
ROLLBACK
```

理解：

```text
一组业务操作要整体成功
```

完成标准：

> 多步操作失败时，不留下半成品数据。

---

## 阶段 9：幂等

重点应用：

```text
闲鱼消息
Webhook
订单事件
```

完成标准：

> 同一条外部消息重复进入，不重复处理。

---

## 阶段 10：并发

学习：

```text
race condition
row lock
optimistic lock
version
状态二次检查
```

完成标准：

> 人工接管和 AI 回复不会互相打架。

---

## 阶段 11：Redis

只在确实出现需要后引入。

学习：

```text
Key
TTL
Cache
Lock
Rate Limit
```

完成标准：

> Redis 只负责临时、高频状态，不成为业务事实来源。

---

# 101. 当前最值得先落地的数据库版本

现在不要一上来设计 15 张表。

第一版只做：

```text
items
conversations
messages
```

就已经足够学习大量核心知识。

---

# 102. 第一版 items

建议：

```text
items

item_id
title
listed_price_cents
sale_status
listing_description
data_source
created_at
updated_at
```

---

# 103. 第一版 conversations

建议：

```text
conversations

conversation_id
channel
buyer_id
item_id
state
created_at
updated_at
```

---

# 104. 第一版 messages

建议：

```text
messages

message_id
external_message_id
conversation_id
sender_type
content
message_type
created_at
raw_payload
```

这三张表已经能覆盖：

```text
商品
会话
消息历史
幂等
人工/AI状态
闲鱼原始消息
```

---

# 105. 暂时不要急着做的内容

当前阶段不需要优先学习：

```text
Kafka
大型分布式数据库
复杂分库分表
数据库集群
读写分离
Sharding
复杂数据仓库
OLAP
Elasticsearch 集群
Kubernetes 数据库运维
数据库内核
复杂复制协议
```

这些不是当前智能客服的瓶颈。

---

# 106. 当前项目真正重要的数据库主线

后面学习时始终围绕这条主线：

```text
业务数据是什么？
↓
谁是 Source of Truth？
↓
应该拆成哪些表？
↓
表之间是什么关系？
↓
需要什么约束？
↓
最常见的查询是什么？
↓
是否需要索引？
↓
多步操作是否需要事务？
↓
重复消息如何保证幂等？
↓
并发情况下会不会写乱？
↓
业务层如何通过 Repository 访问？
↓
数据库结构如何通过 Alembic 演进？
↓
是否真的需要 Redis？
```

只要一直围绕这些问题学习，就不会走偏。

---

# 107. 最终需要真正掌握的核心知识清单

## 必须掌握

```text
关系型数据库基本概念
表设计
主键
外键
一对多
多对多
NULL
数据类型

SELECT
INSERT
UPDATE
DELETE
WHERE
ORDER BY
LIMIT
JOIN

约束
索引

事务
并发基本概念
幂等

PostgreSQL
SQLAlchemy
Alembic

Repository

Source of Truth
```

---

## 需要会用，但不用深入底层

```text
JSONB
Redis
锁
缓存
TTL
数据库备份
数据库连接池
```

---

## 当前不用深入

```text
数据库内核
B+Tree 实现细节
复杂事务隔离理论
分布式一致性算法
数据库集群
分库分表
大型数据仓库
```

这些以后真正遇到规模问题再学习。

---

# 108. 这套数据库知识最终要解决什么问题

不是为了：

```text
简历上写 PostgreSQL
```

而是让你的客服从：

```text
几个 JSON
+
几个 Markdown
+
一个 AI API
```

逐渐变成：

```text
业务数据可靠
消息不重复
状态不混乱
历史可追踪
商品事实明确
订单事实明确
RAG 只负责知识
AI 不编造事实
人工接管不会冲突
数据库结构可持续演进
```

这才是真正的工程价值。

---

# 109. 需要长期记住的 8 条原则

```text
1. PostgreSQL 保存业务事实。

2. Qdrant 保存知识检索索引。

3. Redis 保存临时状态、缓存和锁。

4. LLM 负责理解和表达，不负责创造业务事实。

5. 数据库中“没有数据”不等于“答案是否”。

6. 同一个业务事实必须明确唯一 Source of Truth。

7. 外部消息必须考虑重复投递，所以要设计幂等。

8. 一旦涉及多请求、订单、人工接管，就必须考虑事务和并发。
```

---

# 110. 当前下一步

针对当前智能客服项目，数据库学习不要从 Redis 开始，也不要直接把现有系统全部重构。

最合理的下一步是：

```text
第一步
真正学会 Table / Primary Key / Foreign Key / NULL / 一对多

第二步
练会 SELECT / INSERT / UPDATE / DELETE / JOIN

第三步
根据当前项目亲手设计：
items
conversations
messages

第四步
先画清楚这三张表的字段和关系

第五步
再决定是否将 items.json 迁移到 PostgreSQL

第六步
迁移时引入 Repository + SQLAlchemy

第七步
数据库结构稳定后引入 Alembic

第八步
消息链路继续增加幂等、事务、并发控制

第九步
真正出现缓存或锁需求后再引入 Redis
```

学习数据库的最终判断标准不是：

> “这些名词我都听过。”

而是：

> **当智能客服出现一个数据问题时，你能够判断它应该存在哪里、怎么建模、如何保证不会重复、不会写乱、不会丢失，并能够通过 SQL 把它准确查出来。**

这就是当前项目需要的数据库能力。

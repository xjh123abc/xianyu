# 多轮会话与短期 Memory V1 — 技术文档

## 1. 项目背景

当前主链已经可以完成：

```text
/chat
↓
Router
↓
RAG / MCP / RAG+MCP
↓
DeepSeek
↓
Answer
```

但每次 `/chat` 请求缺少稳定的会话上下文。

下一阶段需要让 ChatService 在处理当前 Query 前，先读取当前会话的历史和状态；回答完成后，再更新当前会话。

---

## 2. 需要什么技术

继续使用：

```text
FastAPI
Python
DeepSeek
现有 RAG
现有 MCP
```

新增一个简单的 Session 管理能力。

核心数据可以先设计为：

```python
sessions = {
    "chat_001": {
        "history": [],
        "state": {
            "order_id": None,
            "last_intent": None
        }
    }
}
```

需要实现 4 个技术点。

### 1. chat_id

`/chat` 请求增加或携带：

```text
chat_id
```

作用：

```text
找到属于当前会话的 History 和 State
```

### 2. History

保存最近几轮：

```text
user message
assistant message
```

调用 DeepSeek 或 Router 时，可以带上必要的历史上下文。

### 3. Session State

保存结构化业务数据，例如：

```text
order_id = TEST1001
last_intent = order_query
```

相比只保存聊天文本，Session State 可以直接告诉系统当前正在处理哪一笔订单。

### 4. Context-aware Router

Router 输入从：

```text
query
```

升级为：

```text
query
+
history
+
session_state
```

例如：

```text
当前 Query：那它一般多久发货？
Session State：order_id = TEST1001
```

系统就可以继续处理当前订单上下文。

---

## 3. 实现什么样的需求

新的主链变成：

```text
POST /chat
↓
读取 chat_id
↓
Session Manager
↓
读取 History + Session State
↓
Router
↓
RAG / MCP / RAG+MCP
↓
DeepSeek
↓
Answer
↓
保存本轮 History
↓
更新 Session State
```

需要实现：

- 同一 `chat_id` 可以连续对话；
- 自动保留当前订单号；
- 后续问题可以省略订单号；
- Router 可以利用上一轮上下文；
- 不同 `chat_id` 之间状态隔离；
- 现有 RAG、MCP 和组合链路继续正常使用。

---

## 4. 最终结果是什么，能否实现需求

最终技术效果：

```text
chat_id = chat_001

用户：查 TEST1001
↓
保存 order_id = TEST1001

用户：那它一般多久发货？
↓
读取 order_id = TEST1001
↓
RAG 查询发货规则

用户：那现在是不是超时了？
↓
读取 TEST1001
↓
MCP 查询订单事实
+
RAG 查询正常时效
↓
DeepSeek 生成最终回答
```

这样可以实现真正的短期多轮会话。

**当前技术完全能够实现这一需求。**

V1 不需要 Redis、向量化聊天记录或长期 Memory。

先用简单 Session Manager 把：

```text
chat_id
history
session_state
context-aware router
```

四个能力打通即可。

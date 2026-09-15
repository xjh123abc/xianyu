# RAG + MCP 组合工作流 V1 — 技术文档

## 1. 背景

当前项目已经有：

```text
FastAPI
ChatService
Router
RAG Pipeline
MCP Client
MCP Server
DeepSeek
```

现有技术已经足够完成下一阶段。

不需要新增 LangGraph、Agent Framework 或其他大型框架。

---

## 2. 要做什么

技术上只需要完成三件事。

### 1. Router 增加组合路由

现有：

```text
rag
mcp
handoff
```

增加：

```text
rag_mcp
```

---

### 2. ChatService 增加组合执行流程

当：

```text
route = rag_mcp
```

时同时执行：

```text
RAGService
+
MCPService
```

得到：

```text
rag_result
mcp_result
```

---

### 3. 增加结果组合

将：

```text
MCP 返回的订单事实
+
RAG 返回的知识库规则
```

交给现有 DeepSeek，生成一个最终回答。

原则：

```text
MCP = 订单真实数据
RAG = 平台规则
LLM = 负责组织语言
```

LLM 不允许自己编造订单状态。

---

## 3. 输出什么结果

最终技术链路：

```text
POST /chat
↓
ChatService
↓
Router
↓
rag_mcp
↓
MCPService + RAGService
↓
订单数据 + 知识库资料
↓
DeepSeek
↓
ChatResponse
```

例如：

```text
TEST1001 现在是什么状态？一般多久发货？
```

最终返回：

```json
{
  "route": "rag_mcp",
  "answer": "订单 TEST1001 当前状态为待发货。根据平台规则，普通商品通常会在付款成功后 24 小时内发出。",
  "sources": [
    {
      "source": "shipping.md"
    }
  ]
}
```

当 `/chat` 能稳定完成：

```text
单独 RAG
单独 MCP
RAG + MCP
```

三种路径，即完成本阶段技术目标。

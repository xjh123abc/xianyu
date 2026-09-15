我需要你对当前整个项目进行一次 **深度 Code Review**。

这次任务的目标不是修改代码，而是找出当前项目中真正存在的架构、逻辑、可靠性和维护性问题。

## 基本原则

本轮只进行 Review。

不要修改正式代码。
不要重构。
不要为了“看起来更好”提出无意义的优化。
不要因为个人代码风格偏好报告问题。

只有能够说明实际影响的问题，才应该进入最终报告。

---

## 第一阶段：理解项目

先完整阅读项目结构和关键代码，不要立即开始找 Bug。

请先梳理：

- 项目的入口
- FastAPI 路由
- Service 层
- RAG 模块
- MCP 模块
- 配置系统
- 数据库 / Qdrant
- 测试代码
- 数据目录
- 主要模型调用

然后画出当前真实调用链。

重点确认类似：

POST /chat

→ ChatService

→ Query / Route 判断

→ Dense Search

→ BM25 Search

→ RRF

→ Reranker

→ Answer Reliability

→ Context Builder

→ LLM / MCP

→ Answer + Sources

不要根据 README 猜测，必须以当前真实代码为准。

如果实际调用链与上述不同，以代码为准。

---

## 第二阶段：检查核心链路

重点 Review 以下部分。

### RAG 检索

检查：

Dense Search  
BM25  
RRF  
Reranker  
TopK  
分数计算  
排序逻辑  
重复结果  
metadata  
chunk / document 对应关系

重点判断：

正确 chunk 是否可能在某一步被错误过滤。

### Answer Reliability

重点检查：

rerank score 和 threshold 是否使用正确。

是否存在：

正确答案已经被检索出来，但是 Reliability 错误拒绝回答。

检查不同模型的 score 是否具有可比性。

### Context Builder

检查：

是否把错误内容传给 LLM。

是否存在：

重复 context  
顺序错误  
metadata 泄露  
空 context  
来源与正文不一致

### LLM Answer

检查：

Prompt 是否能够约束模型只根据知识库回答。

sources 是否真的对应回答依据。

无法回答时是否正确进入 fallback / human handoff。

### MCP

检查：

MCP Server 调用方式  
参数传递  
订单 ID 提取  
错误处理  
超时  
MCP 不可用时系统行为

重点判断：

RAG 和 MCP 的路由是否可能冲突。

### 多轮对话

重点检查：

chat_id  
memory  
历史消息  
item_id  
order_id

确认是否可能发生：

上一轮 ID 丢失  
不同会话上下文串线  
已经识别到 ID，但是下一轮仍然要求用户重新提供  
历史错误信息污染当前回答

---

## 第三阶段：异常场景测试

不要只检查正常流程。

主动检查：

空 query  
不存在的订单  
错误商品 ID  
Qdrant 无结果  
Qdrant 不可用  
Embedding 模型失败  
Reranker 失败  
MCP Server 不可用  
LLM API 失败  
retrieval score 非正常值  
空 context  
重复 chunk  
chat_id 不存在  
连续多轮对话

如果可以通过现有测试或命令验证，请进行验证。

可以运行现有 pytest。

本轮不要修改正式代码。

---

## 第四阶段：代码质量

检查：

重复代码  
废弃代码  
没有使用的模块  
职责混乱  
循环依赖  
过长函数  
硬编码配置  
错误异常捕获  
异常被静默吞掉  
日志缺失  
配置和代码行为不一致

但是：

单纯格式问题不要报告。

---

## 第五阶段：二次反向 Review

完成第一轮后，再进行一次反向检查。

假设你需要故意让这个客服系统产生错误回答。

尝试寻找：

错误路由  
错误召回  
错误重排  
错误 threshold  
错误上下文  
错误 ID  
上下文污染  
错误 fallback  
错误 sources

检查第一轮是否遗漏问题。

---

## 最终报告

不要直接修改代码。

最终只输出 Code Review Report。

按照：

P0：必须立即修复，可能导致系统无法运行、严重错误或数据问题

P1：核心功能错误，会导致错误回答、错误路由、错误状态

P2：中等问题，影响稳定性或维护

P3：低优先级优化

每个问题必须包含：

问题标题

严重级别

文件 + 行号

当前代码行为

触发条件

实际影响

证据

建议修改方向

置信度：高 / 中 / 低

如果只是推测，没有足够证据，请明确写：

「需要进一步验证」

不要把推测写成已经确认的 Bug。

---

最后额外输出：

1. 当前项目最严重的 5 个问题
2. 当前项目最容易出错的调用链
3. 当前项目已经实现得比较正确、不建议修改的部分
4. 建议修复顺序

整个 Review 期间不要修改正式代码。
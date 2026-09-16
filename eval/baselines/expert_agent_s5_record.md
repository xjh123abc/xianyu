# 专家 Agent S5：统一出口并接回主链路

## 本步范围

对照《专家 Agent 需求文档》和《专家 Agent 技术文档》的 S5 四项要求完成实现：

1. 新增 `XianyuExpertOrchestrator`，统一执行规划、专家批处理、依赖调度、结果校验、合并和整条接管。
2. `ChatService` 在已解析商品的闲鱼分支注入并调用编排器；订单、`rag_mcp` 和普通 RAG 分支保持原有边界。无商品的本店规则由服务专家处理。
3. `/chat` 顶层响应补充 `reason`；渠道动作映射在缺少动作且不可回答时稳定转人工，S3 继续负责持久化、通知和一次性买家等待话术。
4. 增加低于频道超时的专家总预算，把剩余时间传给实际 DeepSeek 调用；同步 RAG／模型调用移出事件循环，超时结果不再进入合并。

## 关键行为

- 商品事实、价格和服务事实优先使用结构化卖家数据；模型知识只使用当前商品或本店范围的有效证据。
- 价格仍由唯一的 `PriceAgent` 计算；复合问题按原始任务顺序合并，失败、冲突、缺证据和超时统一返回买家可见的 `稍等我看看`。
- 当前商品上下文继续由 `SessionManager` 保存，支持无 `item_id` 的连续追问；切换商品时交易条件清空。

## 验证

使用 `D:\conda_envs\rag-customer-service\python.exe`：

- S5 新增编排器测试：9 passed。
- 闲鱼测试集：209 passed。
- 闲鱼、API、订单和回归测试全集：268 passed。
- 已将 11 项旧阶段断言迁移到 S5 契约：确定性专家合并不调用旧 `generate_xianyu`，已解析商品不回退到 common 兜底，模型知识改由 `generate_xianyu_expert` 在有效证据范围内生成。
- `compileall` 和 `git diff --check` 通过。

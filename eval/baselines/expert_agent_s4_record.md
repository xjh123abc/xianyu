# 专家 Agent S4：任务规划与商品级上下文记录

实施分支：`expert-agent-s2-price`
S3 基准提交：`f39c976`

## 实施范围

本步骤只完成技术文档 S4：在现有规划器中形成经校验的专家任务，并在既有
`SessionManager.state_json` 中保存商品级追问上下文。

- `build_expert_plan()` 保留现有 `build_question_plan()`；不新建顶层 Router。规则任务是完整性基线：复杂句、无标点长句、条件成交或带价格话题的追问可接收一次注入的规划模型结果，但模型不能替换已确认的规则任务。
- 每个模型任务校验任务 ID、专家、知识范围、原文连续片段、报价金额、运费条件和依赖任务。模型漏题、伪造条件、无效依赖或调用失败时安全回落到规则任务。
- 规划过程对 `IntentRouter` 使用 `allow_ai=False`，避免复合消息的子任务再次触发意图分类模型。
- `xianyu_context` 只保存当前商品 ID、最近价格话题和买家明确选择的运费方案；旧会话读出时补默认值，不改变已有订单字段。切换商品会清空该会话旧的价格话题和运费选择；比较包邮／不包邮不记录为选择。
- 规划提示词明确输出契约和来源约束，要求条件成交的价格任务依赖维修任务。

## S4 直接验收

- A07：`还在吗有没有维修过不包邮最低多少` 形成在售、维修、不包邮最低价三个任务，且不会只保留价格任务。
- A10：本步骤只验证“关键任务不因模型漏项或伪造条件丢失”；统一买家端等待话术、企业微信通知和整条接管属于 S5 编排／渠道接线，尚未在此步骤宣称端到端完成。
- A11：已有最近价格话题时，`那不包邮呢？` 形成买家承担运费的最低价任务；历史回答不提供新的价格政策。
- A12：价格任务保留原始报价和条件，尚未执行价格决策；价格专家继续按 S2 的原始标价和商品政策重新计算，不读取历史报价作为底价。
- A13：切换商品只清空对应会话的 `xianyu_context`，其他会话不受影响。
- 规划模型异常仅记录不含买家内容的诊断并回退规则任务，不造成空计划或渠道副作用。

## 验证结果

```powershell
& "D:\conda_envs\rag-customer-service\python.exe" -m pytest tests\xianyu\test_xianyu_expert_plan.py tests\xianyu\test_intent_router.py tests\xianyu\test_xianyu_expert_agents.py tests\regression\test_stage3_memory.py -q
# 80 passed in 15.10s

& "D:\conda_envs\rag-customer-service\python.exe" -m pytest tests\xianyu -q
# 200 passed in 23.24s

& "D:\conda_envs\rag-customer-service\python.exe" -m pytest tests\api -q
# 10 passed in 21.82s

& "D:\conda_envs\rag-customer-service\python.exe" -m pytest tests\orders -q
# 31 passed in 16.73s

& "D:\conda_envs\rag-customer-service\python.exe" -m pytest tests\regression -q
# 18 passed in 19.59s

& "D:\conda_envs\rag-customer-service\python.exe" -m pytest tests\ingestion -q
# 7 passed in 14.19s

& "D:\conda_envs\rag-customer-service\python.exe" -m pytest tests\retrieval -q
# 47 passed in 21.99s
```

测试总计为 `200 + 10 + 31 + 18 + 7 + 47 = 313 passed`。因当前 Windows
终端对单条命令的 30 秒等待上限，完整 `pytest -q` 按测试目录逐一运行；
`pytest --collect-only -q` 同时确认全套收集数为 313。

## 未完成项

S5 才会新增编排器并把规划结果接回 `ChatService`、`/chat`、HTTP `reason` 与 S3。
本步骤不修改 RAG、LLM 的现有线上调用链、闲鱼消息收发、企业微信通知或任何 HTTP 接口。

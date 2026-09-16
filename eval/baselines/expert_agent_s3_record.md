# 专家 Agent S3：商品／服务专家封装记录

实施分支：`expert-agent-s2-price`  
S2 基准提交：`9bdbc21`

## 实施范围

本步骤只完成技术文档 S3：新增独立的 `ProductAgent`、`ServiceAgent`、共享证据准备方法和专用提示词。

- 专家统一接收 `list[ExpertTask]` 与只读 `ExpertContext`，逐任务返回 `ExpertResult`；不发送闲鱼消息、不通知企业微信、不写会话状态。
- 商品专家先复用 `ItemFactResponder` 的结构化事实读取；字段缺失时，仅可引用 `listing_description` 的原句，或走当前 `item_id` 范围的型号知识检索。
- 服务专家先复用当前商品的 `shipping`／`after_sale` 事实；普通招呼使用固定短句，本店规则才检索 common 范围。
- `XianyuKnowledgeResponder.prepare_evidence()` 复用既有可靠性检查、来源校验和 item/common 范围过滤，只准备证据和不足原因，不生成最终回复。
- 新增规划、商品、服务专用提示词；`DeepSeekGenerator` 新增 `plan_xianyu_questions()` 与 `generate_xianyu_expert()`，复用原模型配置与调用封装。

## S3 直接验收

- A01：商品在售状态由当前商品事实回答，不调用 RAG 或生成模型。
- A08：发货时限与快递分别由当前商品字段回答（48 小时内／中通），不承诺今天或顺丰。
- A09：缺少“测光与手机对比”记录时，商品专家返回 handoff，不从“功能正常”推断。
- A10：任一任务证据不足时返回带缺失字段和内部原因的 handoff 结果，供 S5 统一接管。
- A14：型号知识仅以当前 `item_id` 检索；common 规则不带 item_id，证据不足不调用模型。

S4 才负责将复合句拆成任务并保存追问条件，S5 才会合并专家结果、接回 `/chat` 与 S3。因此本步骤不声称上述复合用例已经完成端到端发送验收。

## 验证结果

```powershell
& "D:\conda_envs\rag-customer-service\python.exe" -m pytest tests\xianyu\test_xianyu_expert_agents.py tests\xianyu\test_xianyu_price_agent.py -q
# 22 passed in 14.97s

& "D:\conda_envs\rag-customer-service\python.exe" -m pytest tests\xianyu tests\api tests\orders tests\regression -q
# 247 passed in 36.56s

& "D:\conda_envs\rag-customer-service\python.exe" -m pytest -q
# 301 passed in 36.39s
```

## 未完成项

S4 的任务规划、连续追问上下文和依赖关系，及 S5 的主链路接线、HTTP `reason` 与渠道接管仍未实现。

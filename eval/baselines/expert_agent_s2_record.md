# 专家 Agent S2：价格专家迁移记录

实施分支：`expert-agent-s2-price`
S1 基准提交：`287cf26`
完成时提交：本分支 S2 最新提交（含 A12 补充验收）

## 实施范围

本步骤只完成需求文档 S2，不接入专家编排层，不替换 `ChatService` 的闲鱼主分支，不新增 HTTP 接口，也不修改 S3 收发消息或企业微信通知。

- 新增 `app/services/xianyu/experts/price_agent.py`：只依据当前商品 `facts`、在售状态和明确价格政策，返回无渠道副作用的 `PriceDecision`。
- 新增最小契约 `app/services/xianyu/experts/contracts.py`：金额均使用整数分，结果携带原价、条件生效价、最低价、买家报价与内部 handoff 原因。
- `ItemFactResponder` 对 PRICE/BARGAIN 只委托 `PriceAgent`，自身不再保留折扣正则、金额计算或报价判断。
- Canon 的原始商品政策文本更新为 `累计最多小刀10元` 与 `不包邮商品价减20元，可叠加小刀`；S1 的旧快照保留不改，供前后比对。
- 路由补充带运费条件的明确报价识别，例如 `1470包邮可以吗？` 必须进入 BARGAIN，而不是被误判为单纯运费问题。

## 已验证的价格职责

- 包邮最低：`1490` 元；不包邮且买家承担运费：`1470` 元。
- 明确出价 `1495` 元时回复接受 `1495`，不主动降到 `1490`。
- 连续议价不累加：先问不包邮最低价返回 `1470` 元，再要求继续优惠时不报价 `1460`，直接返回 handoff 决定。
- `1470包邮`、低于已知下限、自提/面交/拆卖/不要镜头等未授权条件、政策缺失或冲突均返回 handoff 决定。
- 已售商品只说明已售，不接受报价。
- 每次均从原始标价重新计算；没有把历史 AI 回复当作新的价格政策。
- PRICE/BARGAIN 结构化价格回答不调用 RAG，也不调用自由生成模型；A12 连续场景已显式断言该行为。

## 验证结果

```powershell
& "D:\conda_envs\rag-customer-service\python.exe" -m pytest tests\xianyu\test_xianyu_price_agent.py tests\xianyu\test_xianyu_structured_facts.py tests\xianyu\test_xianyu_complex_buyer_messages.py tests\xianyu\test_intent_router.py -q
# 103 passed in 17.89s

& "D:\conda_envs\rag-customer-service\python.exe" -m pytest tests\xianyu tests\api tests\orders tests\regression -q
# 239 passed in 32.85s

& "D:\conda_envs\rag-customer-service\python.exe" -m pytest -q
# 293 passed in 37.75s
```

## 未完成项

S3 才开始商品/服务专家与范围化知识证据准备；S4 才记录商品级价格话题和连续追问条件；S5 才将专家编排层接入 `ChatService` 并传递 handoff reason 到 HTTP/S3。当前仍保持旧闲鱼回答主路径，确保 S2 可以独立回归。

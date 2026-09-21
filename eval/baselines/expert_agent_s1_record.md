# 专家 Agent S1：固定基准与接入点记录

记录时间：2026-09-15T11:24:07+08:00
实施分支：`expert-agent-s1-baseline`
实际代码 HEAD：`50fffe6d998ad7555192aa33d75a0f22510ea4c9`

## 基线核对

需求与技术文档声明的代码基线为 `385ee01e8f9007ec34530c7a06ce9ec55ece6b9c`。
该对象不在当前本地仓库中：`git merge-base --is-ancestor` 退出码为 128，Git 报告该提交不是有效对象。因此本次不能伪造该 SHA 到当前 HEAD 的差异；后续实施以本记录的实际 HEAD 为可复现基线，并在取得该对象后再补充历史差异核对。

当前 HEAD 的最近提交为：

```text
50fffe6 feat: fix rag readiness and corpus isolation
4b5b77b feat: reply
e3bcb3e feat: xianyu v2 s3
```

## 固定商品数据

Canon 商品原始快照保存于 [expert_agent_s1_canon_ftb_001.json](expert_agent_s1_canon_ftb_001.json)。

- 来源：`data/xianyu/items.json`
- 来源文件 SHA-256：`972499F9855AB8D5262222740EB61A48382C93F4A1B9ECB56513E81AC1810415`
- 商品：`CANON_FTB_001`，标价 `150000` 分，在售，允许议价
- 当前原始政策文本：`小刀10元，1490元可以拍`；`不包邮售价减20 1480元`

S1 不修改该商品数据。S2 改写政策文本或价格实现前，必须以本快照比对回归结果。

## 指定基准测试

执行命令：

```powershell
& "D:\conda_envs\rag-customer-service\python.exe" -m pytest tests\xianyu tests\api tests\orders tests\regression -q
```

结果：`223 passed in 32.55s`。该命令内没有既有失败，也没有为专家 Agent 删除、跳过或放宽任何断言。

本记录不把未运行的真实渠道联调或全量模型推理记为通过；它们属于 S6 的受控验证范围。

## 已确认的接入点与提前返回

`POST /chat` 继续经 `app/api/chat.py` 调用 `ChatService.chat_async()`。S5 唯一的业务接入点应位于 `ChatService._chat_async_locked()` 已完成商品解析和订单分流之后；订单、`rag_mcp`、普通 RAG 以及商品身份解析失败不能被专家层截走。

当前必须在 S5 替换或收敛的旧闲鱼回答出口如下：

| 位置 | 当前行为 | S5 约束 |
|---|---|---|
| `app/services/chat_service.py:180` | `ItemContextResolver` 返回商品冲突／缺失时提前返回 | 保留，不调用专家猜测商品 |
| `app/services/chat_service.py:219-235` | 简单单意图直接事实回答；其他有商品问题交 `handle_item()` | 由编排器统一处理，单请求只能走一套回答逻辑 |
| `app/services/chat_service.py:244-265` | 无商品的商品字段／本店规则问题走澄清或 `handle_common()` | 由服务专家处理已确认的本店规则；本期澄清仍转人工 |
| `app/services/chat_service.py:294-304` | `ChatService` 在最终响应后只写一次会话历史 | 保留；专家不得自行写历史 |
| `app/services/xianyu/knowledge_responder.py:75-81` | `BARGAIN` 对整条消息直接返回价格事实结果 | S2/S5 必须移除该整条提前返回，否则“维修＋议价”会漏答维修任务 |

当前还确认以下后续任务，S1 仅记录、不修复：

1. `app/api/chat.py:82-101` 的顶层 `Response` 没有 `reason`，现有 handoff 内部原因会在 HTTP 序列化时丢失；S5 需补充兼容字段和 API 回归测试。
2. `app/channels/xianyu/action_mapper.py:51-54` 在无 `action` 且 `can_answer=False` 的 fail-closed 分支引用未定义变量 `clarify`；S1 已静态确认，S5 按技术文档改为稳定人工接管并补测试。
3. `app/services/xianyu/item_fact_responder.py` 仍承载 PRICE/BARGAIN 计算；S2 应迁移到唯一的 `PriceAgent`，不保留两套长期并行规则。

## S1 实施边界结论

本步骤未修改业务回答路径、商品 JSON、价格规则、会话状态、`/chat` 契约、S3 收发消息或企业微信通知。下一步仅进入 S2：先建立可单测的价格专家，并继续使用本快照和本记录的固定基准。

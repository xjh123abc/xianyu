# 闲鱼 AI 客服下一阶段实施文档

## 一、阶段目标

当前 50 条测试中，大部分问题没有进入正常回复。

下一阶段不继续优化 RAG、Embedding、Rerank，也不更换模型。

本阶段只解决 3 个核心问题：

1. **买家说法不同，但系统无法识别成同一个意图。**
2. **商品数据不完整，系统即使知道买家在问什么，也没有事实可以回答。**
3. **`reply / clarify / handoff` 的判断逻辑不清晰。**

最终目标：

> 买家问题 → 识别意图 → 查询商品事实 → 决定回复 / 澄清 / 转人工。

---

# 二、下一阶段主流程

```text
买家消息
   ↓
Intent Router（识别买家想问什么）
   ↓
读取当前商品数据
   ↓
判断数据是否足够
   ↓
┌─────────────┬─────────────┬─────────────┐
│             │             │
有可靠数据    问题不明确      数据缺失
│             │             │
reply         clarify       handoff
│
固定模板 / 必要时 AI
```

---

# 三、第一步：先修基础逻辑

## 1. 修复价格显示

如果字段：

```json
"listed_price_cents": 128000
```

表示“分”，则最终价格必须转换：

```python
price_yuan = listed_price_cents / 100
```

结果：

```text
¥1280.00
```

不能直接显示：

```text
¥128000.00
```

---

## 2. 重新定义 3 个 action

### reply

系统知道买家在问什么，并且有可靠数据。

```text
买家：这个还在吗？
sale_status = listed
→ reply
```

### clarify

系统无法判断买家到底在问什么，或者缺少必须由买家补充的信息。

```text
买家：这个怎么样？
→ clarify
```

### handoff

系统知道买家在问什么，但商品数据中没有答案，需要卖家确认。

```text
买家：以前维修过吗？
repair_history = unknown
→ handoff
```

---

# 四、第二步：扩充商品数据

当前商品数据不能只保存：

```text
标题
价格
出售状态
```

至少增加下面这些字段：

```json
{
  "item_id": "CANON_FTB_001",

  "title": "Canon FTb 胶片相机",

  "listed_price_cents": 128000,

  "sale_status": "listed",

  "negotiation": "firm",

  "condition": {
    "summary": "正常使用成色",
    "scratches": "机身底部有轻微划痕",
    "dents": "none"
  },

  "function": {
    "overall": "working",
    "shutter": "working"
  },

  "history": {
    "repair_history": "unknown",
    "drop_history": "unknown",
    "disassembly_history": "unknown"
  },

  "accessories": [
    "机身",
    "镜头",
    "镜头盖"
  ],

  "shipping": {
    "ship_from": "待填写",
    "carrier": "待填写",
    "shipping_fee": "待填写",
    "dispatch_time": "待填写"
  },

  "return_policy": "manual_review"
}
```

## 数据原则

对于不能确认的事实，不要留空后让 AI 猜。

统一使用：

```text
yes
no
unknown
```

例如：

```json
"repair_history": "unknown"
```

表示：

> 当前没有可靠信息，需要人工确认。

---

# 五、第三步：建立 Intent Router

第一版只做最常见的 14 个 Intent：

```text
AVAILABILITY      是否还在售
PRICE             价格
BARGAIN           议价
CONDITION         成色
DEFECT            瑕疵
REPAIR_HISTORY    维修 / 拆修 / 摔落历史
FUNCTION          功能状态
ACCESSORIES       配件
SHIPPING_TIME     发货时间
SHIPPING_FEE      包邮 / 运费
PRODUCT_INFO      型号等商品信息
AFTER_SALE        退货 / 售后
GREETING          打招呼
OTHER             其他
```

每个 Intent 需要配置 4 样东西：

```text
1. 常见问法
2. 需要读取哪些商品字段
3. 数据存在时怎么回复
4. 数据缺失时怎么处理
```

例如：

```text
Intent：AVAILABILITY

常见问法：
- 这个还在吗？
- 还没出吧？
- 还能拍吗？
- 卖了吗？
- 还有货吗？

读取字段：
sale_status

判断：
listed → reply
sold → reply
unknown → handoff
```

---

# 六、Intent Router 的第一版实现方式

不要一开始全部交给 LLM。

先使用：

```text
规则 / 关键词 / 常见表达
        ↓
匹配成功
        ↓
直接得到 Intent
```

没有匹配成功时，再让 AI 做语义分类：

```text
规则没有识别
        ↓
AI 判断 Intent
        ↓
只返回分类结果
```

例如：

```json
{
  "intent": "BARGAIN"
}
```

AI 只负责：

> “买家在问什么？”

AI 不负责：

> “最低多少钱？”

价格和交易规则必须读取你的商品数据。

---

# 七、第四步：重新跑同一批 50 条测试

完成以上改造后，不换题。

继续使用原来的 50 条测试。

对比：

```text
改造前
reply = 4 / 50

改造后
reply = ?
clarify = ?
handoff = ?
```

测试结果至少增加：

```json
{
  "question": "还没出吧？",

  "intent": "AVAILABILITY",

  "item_id": "CANON_FTB_001",

  "required_fields": [
    "sale_status"
  ],

  "facts": {
    "sale_status": "listed"
  },

  "action": "reply",

  "answer": "还在的，目前可以正常拍。"
}
```

这样以后出现错误，可以直接判断问题在：

```text
Intent 识别
↓
商品数据
↓
决策逻辑
↓
回复模板
```

而不是只看到最终答案。

---

# 八、本阶段完成标准

本阶段完成以后，至少满足：

- 同一个意思换不同说法，能够识别成同一个 Intent。
- 价格显示正确。
- 已有商品事实的问题可以自动回复。
- 商品事实缺失时不会让 AI 编造。
- `clarify` 只用于真正需要买家补充的问题。
- 已知问题但缺少商品事实时进入 `handoff`。
- 原来的 50 条测试可以自动重复运行。
- 可以看到每条消息识别出的 Intent 和使用的商品事实。

---

# 九、本阶段暂时不要做

暂时不要继续优化：

```text
Embedding
Dense Search
BM25
RRF
Reranker
Reliability Threshold
更换 LLM
```

这些不是当前 50 条测试失败的主要原因。

当前唯一主线：

```text
Intent Router
+
商品事实数据
+
客服决策规则
```

这三部分稳定以后，再决定哪些问题真正需要进入 RAG / AI。

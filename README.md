# E-commerce AI Customer Service

基于 RAG 的电商智能客服项目骨架。

## 目录

- `app/`：应用 API、数据摄取、检索、RAG 和生成逻辑
- `config/`：配置
- `data/raw/`：原始业务文档
- `eval/`：评测数据与评测脚本
- `mcp_servers/`：独立 MCP Server 入口与模拟订单工具
- `scripts/`：MCP 与统一 `/chat` 端到端验收脚本
- `tests/`：测试

## 快速开始

```bash
python -m venv .venv
pip install -r requirements.txt
copy .env.example .env
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

### RAG 启动前置条件与就绪检查

项目只从本地目录加载 embedding 和 reranker 模型，不会在请求期间下载模型。复制示例配置后，必须在 `.env` 中将以下两项替换为实际存在的本地模型目录：

```dotenv
EMBEDDING_MODEL_PATH=/absolute/path/to/local-embedding-model
RERANKER_MODEL_PATH=/absolute/path/to/local-reranker-model
```

`GET /health` 仅表示 FastAPI 进程已启动；`GET /ready` 会实际加载普通 RAG 使用的两个本地模型，并验证 Qdrant collection 可访问。若模型路径错误、模型无法加载、Qdrant 不可达或 collection 尚未摄取，`/ready` 返回 HTTP 503；只有返回 `{"status":"ready","rag":"ready"}` 才表示普通 RAG 链路可用。

首次启动前需要先完成知识库摄取，并确认 `/ready` 成功：

```powershell
& "D:\conda_envs\rag-customer-service\python.exe" -m app.ingestion.pipeline
Invoke-RestMethod -Uri "http://127.0.0.1:8000/ready"
```

当前项目保留原电商 RAG/MCP 链路，并增加了本地闲鱼示例商品问答。

## 统一商品问答

三件示例商品位于 `data/xianyu/`。先同步独立的闲鱼知识域：

```powershell
& "D:\conda_envs\rag-customer-service\python.exe" -m app.ingestion.pipeline --scenario xianyu
```

运行自动化测试和真实 MCP 协议 smoke（覆盖三件商品及不存在编号）：

```powershell
& "D:\conda_envs\rag-customer-service\python.exe" -m pytest
& "D:\conda_envs\rag-customer-service\python.exe" -m scripts.smoke_order_mcp
```

调用已启动服务；商品上下文由 `chat_id` 和 `item_id` 确定，不需要模式参数：

```powershell
$body = @{
  query = "这个商品多少钱，带哪些配件？"
  chat_id = "qa_item_001"
  item_id = "DEMO_ITEM_001"
} | ConvertTo-Json
Invoke-RestMethod -Uri "http://127.0.0.1:8000/chat" -Method Post `
  -ContentType "application/json; charset=utf-8" `
  -Body ([System.Text.Encoding]::UTF8.GetBytes($body))
```

服务启动后，运行 A01—A14 统一验收。脚本只调用正在运行的真实 `/chat`，不替换 ChatService、MCP、检索、Reranker 或模型，不读取 Cookie、不发闲鱼消息，也不修改商品和知识库。结果写入 `eval/results/`；标记为“阻塞”的开放式回答必须对照真实资料人工复核，不能当作自动通过：

```powershell
& "D:\conda_envs\rag-customer-service\python.exe" -m scripts.smoke_unified_chat --base-url http://127.0.0.1:8000
```

## 专家 Agent S6 验收与启用

真实自动发送默认保持暂停。先运行代码测试和 60 条真实 `/chat` 批测；批测结果包含每题的独立 `chat_id`、确定性任务规划、答案、动作、来源、内部接管原因和耗时：

```powershell
& "D:\conda_envs\rag-customer-service\python.exe" -m pytest tests/xianyu tests/api tests/orders tests/regression -q
& "D:\conda_envs\rag-customer-service\python.exe" -m pytest -q
& "D:\conda_envs\rag-customer-service\python.exe" -m eval.batch_chat_test `
  --base-url http://127.0.0.1:8000 `
  --item-id CANON_FTB_001 `
  --output eval/results/expert_agent_s6_batch.json
```

真实渠道验收只能在明确的测试会话中临时开启。必须同时指定账号、唯一会话、允许的买家原文和处理条数；其他会话及其他文本均被忽略，进程退出时账号自动暂停：

```powershell
& "D:\conda_envs\rag-customer-service\python.exe" scripts/run_xianyu_stage3.py `
  --reference-root .runtime/xianyu-template `
  --db logs/xianyu_stage3.sqlite3 `
  --log-file logs/xianyu_s6_acceptance.log `
  --account TEST_SELLER `
  --controlled-acceptance `
  --acceptance-chat TEST_CHAT `
  --acceptance-query "包邮最低多少？" `
  --acceptance-max-messages 1
```

正式常驻前，复制 [S6 验收报告模板](eval/baselines/expert_agent_s6_acceptance.template.json)，填写当前 Git 提交及实际结果。只有代码测试、60 条批测、A01—A18、真实模型、真实渠道均为 `passed` 且 `approved_for_auto_send=true` 时，运行器才接受该报告：

```powershell
& "D:\conda_envs\rag-customer-service\python.exe" scripts/run_xianyu_stage3.py `
  --reference-root .runtime/xianyu-template `
  --db logs/xianyu_stage3.sqlite3 `
  --log-file logs/xianyu_stage3.log `
  --acceptance-report logs/expert_agent_s6_acceptance.json `
  --enable
```

可在 `.env` 设置 `XIANYU_EXPERT_BUDGET_SECONDS=25`；该值应小于渠道 HTTP 超时。Mock、真实模型和真实渠道结果必须分开记录，任一项未运行都不能批准正式自动发送。

## MCP 模拟订单

当前 MCP 部分提供本地模拟订单，不接真实数据库、淘宝或物流平台；订单查询已经通过简单分流接回 `/chat`，普通知识问题仍走原有 RAG。

使用目标 Conda 环境执行真实协议 smoke test：

```powershell
& "D:\conda_envs\rag-customer-service\python.exe" -m scripts.smoke_order_mcp
```

独立启动 MCP Server（stdio 模式，通常由 MCP Client 启动）：

```powershell
& "D:\conda_envs\rag-customer-service\python.exe" mcp_servers\order_server.py
```

通过 `/chat` 查询模拟订单：

```powershell
$body = @{
  query = "帮我查订单 TEST1001 发货了吗？"
  chat_id = "qa_order_001"
} | ConvertTo-Json
Invoke-RestMethod -Uri "http://127.0.0.1:8000/chat" -Method Post `
  -ContentType "application/json; charset=utf-8" `
  -Body ([System.Text.Encoding]::UTF8.GetBytes($body))
```

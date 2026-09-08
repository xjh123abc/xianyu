# E-commerce AI Customer Service

基于 RAG 的电商智能客服项目骨架。

## 目录

- `app/`：应用 API、数据摄取、检索、RAG 和生成逻辑
- `config/`：配置
- `data/raw/`：原始业务文档
- `eval/`：评测数据与评测脚本
- `mcp_servers/`：独立 MCP Server 入口与模拟订单工具
- `scripts/`：MCP 端到端 smoke 脚本
- `tests/`：测试

## 快速开始

```bash
python -m venv .venv
pip install -r requirements.txt
copy .env.example .env
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

当前文件为项目骨架；请继续实现各模块中的业务逻辑，并将真实政策文档放入 `data/raw/`。

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
$body = @{ query = "帮我查订单 TEST1001 发货了吗？" } | ConvertTo-Json
Invoke-RestMethod -Uri "http://127.0.0.1:8000/chat" -Method Post `
  -ContentType "application/json; charset=utf-8" `
  -Body ([System.Text.Encoding]::UTF8.GetBytes($body))
```

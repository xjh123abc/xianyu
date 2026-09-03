# E-commerce AI Customer Service

基于 RAG 的电商智能客服项目骨架。

## 目录

- `app/`：应用 API、数据摄取、检索、RAG 和生成逻辑
- `config/`：配置
- `data/raw/`：原始业务文档
- `eval/`：评测数据与评测脚本
- `tests/`：测试

## 快速开始

```bash
python -m venv .venv
pip install -r requirements.txt
copy .env.example .env
uvicorn app.main:app --reload
```

当前文件为项目骨架；请继续实现各模块中的业务逻辑，并将真实政策文档放入 `data/raw/`。


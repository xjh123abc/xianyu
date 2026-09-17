# 电商 RAG 测试数据

这是一套专门用于练习 RAG 的模拟电商知识库，不是真实平台政策。

## 目录
- knowledge_base/: 12 个 Markdown 知识文件，可直接作为 Loader 输入。
- eval_questions.json: 15 条人工测试问题，每条包含预期命中的文件和关键词。

## 建议测试步骤
1. 把 knowledge_base/ 目录放到你项目的数据目录中。
2. 重新执行 Loader → Chunk → Embedding → Point → Qdrant 写入。
3. 用 eval_questions.json 里的 query 调用你的 VectorSearch.search(query, top_k=5)。
4. 查看 Top 1 / Top 3 / Top 5 返回结果。
5. 如果 expected_file 对应的知识出现在 Top 3，说明这一题的 Dense Retrieval 基本命中。

## 注意
为了后续测试 BM25，本数据中故意加入了 E1001、E2001、E3001 等错误码。
为了测试 Dense Retrieval，也加入了“退款多久能到账”与“我的钱什么时候能收到”这类字面不同、语义相近的问题。

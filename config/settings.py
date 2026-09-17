"""Project settings loaded from environment variables."""

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict

    class Settings(BaseSettings):
        model_config = SettingsConfigDict(
            env_file=".env",
            env_file_encoding="utf-8",
            extra="ignore",
        )

        deepseek_api_key: str = ""
        deepseek_base_url: str = "https://api.deepseek.com"
        deepseek_model: str = "deepseek-chat"
        deepseek_timeout: float = 60.0
        deepseek_temperature: float = 0.2
        deepseek_max_tokens: int = 512
        qdrant_url: str = "http://localhost:6333"
        qdrant_collection: str = "ecommerce_documents"
        knowledge_base_path: str = "data/ecommerce/knowledge"
        knowledge_base_chunk_size: int = 500
        embedding_model_path: str = ""
        reranker_model_path: str = ""
        dense_top_k: int = 5
        bm25_top_k: int = 5
        rrf_k: int = 60
        reranker_top_k: int = 5
        rag_context_max_chunks: int = 3
        rag_context_score_gap: float = 0.15
        ingestion_manifest_path: str = ".rag_ingestion_manifest.json"
        answer_reliability_threshold: float = 5.3
        answer_reliability_model_id: str = "Qwen3-Reranker-0.6B"
        session_database_path: str = "chat_sessions.sqlite3"
        session_ttl_seconds: int = 3600
        session_max_count: int = 10000
        session_lock_timeout_seconds: float = 60.0
        knowledge_corpus_id: str = "ecommerce"
        xianyu_corpus_id: str = "xianyu"
        xianyu_items_path: str = "data/xianyu/items.json"
        xianyu_knowledge_base_path: str = "data/xianyu/knowledge"
        xianyu_qdrant_collection: str = "xianyu_documents"
        xianyu_ingestion_manifest_path: str = ".xianyu_ingestion_manifest.json"
        xianyu_chat_api_base_url: str = "http://127.0.0.1:8000"
        xianyu_chat_api_timeout_seconds: float = 30.0
        xianyu_expert_budget_seconds: float = 25.0
        wecom_webhook_url: str = ""




    settings = Settings()
except ImportError:  # Keeps the scaffold usable before dependencies are installed.


    settings = None

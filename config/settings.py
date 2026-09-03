"""Project settings loaded from environment variables."""

try:
    from pydantic_settings import BaseSettings

    class Settings(BaseSettings):
        deepseek_api_key: str = ""
        deepseek_base_url: str = "https://api.deepseek.com"
        deepseek_model: str = "deepseek-chat"
        qdrant_url: str = "http://localhost:6333"
        qdrant_collection: str = "ecommerce_documents"
        embedding_model_path: str = (
            r"D:\local_models\huggingface\hub\models--Qwen--Qwen3-Embedding-0.6B"
            r"\snapshots\97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
        )

    settings = Settings()
except ImportError:  # Keeps the scaffold usable before dependencies are installed.
    settings = None

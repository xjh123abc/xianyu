from app.ingestion.chunker import chunk_document
from app.ingestion.loader import load_txt
from app.services.embedding_service import EmbeddingService
from config.settings import settings


def test_chunks_generate_matching_local_embeddings() -> None:
    document = load_txt("data/raw/ingestion_test.txt")
    chunks = chunk_document(document, chunk_size=100)
    service = EmbeddingService()

    vectors = service.embed_chunks(chunks)

    assert service.model_path.as_posix().endswith(
        "models--Qwen--Qwen3-Embedding-0.6B/snapshots/"
        "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
    )
    assert settings is not None
    assert str(service.model_path) == settings.embedding_model_path
    assert len(vectors) == len(chunks)
    assert len(vectors) > 1
    assert all(isinstance(vector, list) and vector for vector in vectors)
    assert len({len(vector) for vector in vectors}) == 1
    assert all(isinstance(value, float) for vector in vectors for value in vector)

from app.ingestion.chunker import chunk_document
from app.ingestion.loader import load_txt
from app.services import embedding_service as embedding_module
from app.services.embedding_service import EmbeddingService
from config.settings import settings


class FakeEmbeddingModel:
    def encode(self, values, **kwargs):
        items = values if isinstance(values, list) else [values]
        return FakeEncoded([[float(index + 1), 0.5] for index, _ in enumerate(items)])


class FakeEncoded(list):
    def tolist(self):
        return list(self)


def test_chunks_generate_matching_local_embeddings(monkeypatch) -> None:
    document = load_txt("tests/fixtures/ingestion_test.txt")
    chunks = chunk_document(document, chunk_size=100)
    model_path = embedding_module.Path(__file__).parent
    monkeypatch.setattr(embedding_module.settings, "embedding_model_path", str(model_path))
    monkeypatch.setattr(embedding_module, "_load_model", lambda _: FakeEmbeddingModel())
    service = EmbeddingService()

    vectors = service.embed_chunks(chunks)

    assert settings is not None
    assert service.model_path == model_path
    assert len(vectors) == len(chunks)
    assert len(vectors) > 1
    assert all(isinstance(vector, list) and vector for vector in vectors)
    assert len({len(vector) for vector in vectors}) == 1
    assert all(isinstance(value, float) for vector in vectors for value in vector)

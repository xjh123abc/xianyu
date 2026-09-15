from fastapi.testclient import TestClient

from app import main
from app.main import app


client = TestClient(app)


def test_health_returns_200() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready_returns_503_when_rag_dependencies_are_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(
        main.chat_service.rag_service,
        "warm_up",
        lambda: (_ for _ in ()).throw(FileNotFoundError("embedding model missing")),
    )

    response = client.get("/ready")

    assert response.status_code == 503
    assert response.json() == {
        "detail": {"status": "not_ready", "component": "rag"}
    }


def test_ready_verifies_the_shared_rag_service(monkeypatch) -> None:
    class ReadyVectorSearch:
        def __init__(self) -> None:
            self.verified = False

        def verify_collection(self) -> None:
            self.verified = True

    vector_search = ReadyVectorSearch()
    monkeypatch.setattr(main.chat_service.rag_service, "warm_up", lambda: None)
    monkeypatch.setattr(main.chat_service.rag_service, "vector_search", vector_search)

    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ready", "rag": "ready"}
    assert vector_search.verified is True

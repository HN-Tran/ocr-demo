from __future__ import annotations

from starlette.testclient import TestClient

from app.main import _create_ocr_app
from tests.test_main import _settings


def test_inference_providers_endpoint() -> None:
    client = TestClient(_create_ocr_app(settings=_settings()))
    response = client.get("/api/inference-providers")
    assert response.status_code == 200
    payload = response.json()
    assert payload["default_provider"] == "ollama"
    assert any(entry["id"] == "ollama" for entry in payload["providers"])

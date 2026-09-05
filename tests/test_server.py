"""Integration test for the /generate endpoint.

Loads the real Qwen/Qwen2.5-0.5B-Instruct model (no mocking) via the
FastAPI app's actual startup lifecycle, so this is slow and marked
accordingly. The client fixture is module-scoped so the model is loaded
exactly once for the whole test module, even if more tests are added
later.
"""

import pytest
from fastapi.testclient import TestClient

from src.server.main import app

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_generate_returns_nonempty_completion(client):
    response = client.post(
        "/generate",
        json={"prompt": "What is the capital of France?"},
    )

    assert response.status_code == 200

    body = response.json()
    assert isinstance(body["completion"], str)
    assert body["completion"].strip() != ""
    assert body["input_tokens"] > 0
    assert body["output_tokens"] > 0

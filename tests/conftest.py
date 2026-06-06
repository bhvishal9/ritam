from collections.abc import Generator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from ritam.main import app
from tests.fakes import FakeLlmClient, NoCallLlmClient


@pytest.fixture(scope="function")
def client() -> Generator[TestClient, Any]:
    """
    A fixture to provide a FastAPI TestClient for API testing.
    """
    with TestClient(app) as c:
        yield c


@pytest.fixture
def fake_llm_client() -> FakeLlmClient:
    """
    A fixture to provide a fake LLM client for testing.
    """
    return FakeLlmClient()


@pytest.fixture
def no_call_llm_client() -> NoCallLlmClient:
    """
    A fixture to provide a no-call LLM client for testing.
    """
    return NoCallLlmClient()

import pytest
from _pytest.monkeypatch import MonkeyPatch
from pydantic import ValidationError

from ritam.config.settings import Settings, VectorStoreType, _is_local

_QDRANT_ENV = ["VECTOR_STORE", "QDRANT_URL", "QDRANT_API_KEY"]


@pytest.fixture(autouse=True)
def base_env(monkeypatch: MonkeyPatch) -> None:
    """Required fields present; Qdrant vars cleared so each test sets its own."""
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("SOURCE_URI", "file:///tmp/docs")
    for var in _QDRANT_ENV:
        monkeypatch.delenv(var, raising=False)


class TestIsLocal:
    def test_localhost_is_local(self) -> None:
        assert _is_local("http://localhost:6333") is True

    def test_loopback_ipv4_is_local(self) -> None:
        assert _is_local("http://127.0.0.1:6333") is True

    def test_remote_cloud_is_not_local(self) -> None:
        assert _is_local("https://abc.cloud.qdrant.io:6333") is False


class TestSettingsValidation:
    def test_file_store_needs_no_qdrant_config(self, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setenv("VECTOR_STORE", "file")
        settings = Settings(_env_file=None)
        assert settings.vector_store == VectorStoreType.FILE
        assert settings.qdrant_client_url is None

    def test_qdrant_without_url_raises(self, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setenv("VECTOR_STORE", "qdrant")
        with pytest.raises(ValidationError, match="QDRANT_URL"):
            Settings(_env_file=None)

    def test_qdrant_local_needs_no_key(self, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setenv("VECTOR_STORE", "qdrant")
        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
        settings = Settings(_env_file=None)
        assert settings.qdrant_api_key is None

    def test_qdrant_remote_without_key_raises(self, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setenv("VECTOR_STORE", "qdrant")
        monkeypatch.setenv("QDRANT_URL", "https://abc.cloud.qdrant.io:6333")
        with pytest.raises(ValidationError, match="QDRANT_API_KEY"):
            Settings(_env_file=None)

    def test_qdrant_remote_with_key_ok(self, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setenv("VECTOR_STORE", "qdrant")
        monkeypatch.setenv("QDRANT_URL", "https://abc.cloud.qdrant.io:6333")
        monkeypatch.setenv("QDRANT_API_KEY", "secret")
        settings = Settings(_env_file=None)
        assert settings.qdrant_api_key == "secret"

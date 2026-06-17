import pytest
from pydantic import BaseModel, ValidationError
from pytest_mock import MockerFixture

from ritam.api.dependencies import get_llm_client, get_vector_store_client
from ritam.api.exceptions import CustomException


class _RequiresSourceUri(BaseModel):
    source_uri: str


def _missing_field_error() -> ValidationError:
    """A real ValidationError for a missing required field, as Settings raises."""
    try:
        _RequiresSourceUri()  # type: ignore[call-arg]
    except ValidationError as err:
        return err
    raise AssertionError("expected a ValidationError")


class TestConfigurationErrorMapping:
    def test_llm_client_config_error_names_the_field(
        self, mocker: MockerFixture
    ) -> None:
        mocker.patch(
            "ritam.api.dependencies.create_llm_client",
            side_effect=_missing_field_error(),
        )

        with pytest.raises(CustomException) as exc_info:
            get_llm_client()

        assert exc_info.value.status_code == 500
        assert "Configuration error" in exc_info.value.message
        # The failing field is surfaced, not hidden behind a generic message...
        assert "source_uri" in exc_info.value.message
        # ...and it is not mislabelled as an LLM-specific failure.
        assert "LLM" not in exc_info.value.message

    def test_vector_store_client_config_error_names_the_field(
        self, mocker: MockerFixture
    ) -> None:
        mocker.patch(
            "ritam.api.dependencies.create_vector_store_client",
            side_effect=_missing_field_error(),
        )

        with pytest.raises(CustomException) as exc_info:
            get_vector_store_client()

        assert exc_info.value.status_code == 500
        assert "Configuration error" in exc_info.value.message
        assert "source_uri" in exc_info.value.message

import pytest
from pytest_mock import MockerFixture

from ritam.core.factories import (
    create_document_source_client,
    create_vector_store_client,
)
from ritam.document_source.local_document_source import LocalDocumentSource


class TestFactories:
    def test_create_vector_store_client_raises_for_unsupported_type(
        self, mocker: MockerFixture
    ) -> None:
        mock_settings = mocker.MagicMock()
        mock_settings.vector_store = "unsupported"
        mocker.patch("ritam.core.factories.get_settings", return_value=mock_settings)

        with pytest.raises(NotImplementedError):
            create_vector_store_client()

    def test_create_document_source_client_raises_when_source_uri_missing(
        self, mocker: MockerFixture
    ) -> None:
        mock_settings = mocker.MagicMock()
        mock_settings.source_uri = None
        mocker.patch("ritam.core.factories.get_settings", return_value=mock_settings)

        with pytest.raises(ValueError, match="SOURCE_URI is required"):
            create_document_source_client()

    def test_create_document_source_client_returns_local_for_file_uri(
        self, mocker: MockerFixture
    ) -> None:
        mock_settings = mocker.MagicMock()
        mock_settings.source_uri = "file:///tmp/docs"
        mocker.patch("ritam.core.factories.get_settings", return_value=mock_settings)

        assert isinstance(create_document_source_client(), LocalDocumentSource)

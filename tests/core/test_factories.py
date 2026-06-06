import pytest
from pytest_mock import MockerFixture

from ritam.core.factories import create_vector_store_client


class TestFactories:
    def test_create_vector_store_client_raises_for_unsupported_type(
        self, mocker: MockerFixture
    ) -> None:
        mock_settings = mocker.MagicMock()
        mock_settings.vector_store = "unsupported"
        mocker.patch("ritam.core.factories.get_settings", return_value=mock_settings)

        with pytest.raises(NotImplementedError):
            create_vector_store_client()

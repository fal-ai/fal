import pytest
from pydantic import BaseModel, ValidationError

from fal.toolkit.types import MAX_DATA_URI_LENGTH, MAX_HTTPS_URL_LENGTH, FileInput


class DummyModel(BaseModel):
    url: FileInput


class TestFileInput:
    def test_valid_https_urls(self):
        # Test basic HTTPS URL
        model = DummyModel(url="https://example.com")
        assert model.url == "https://example.com"

        # Test HTTPS URL with path
        model = DummyModel(url="https://example.com/path/to/resource")
        assert model.url == "https://example.com/path/to/resource"

        # Test HTTPS URL with query parameters
        model = DummyModel(url="https://example.com/search?q=test&page=1")
        assert model.url == "https://example.com/search?q=test&page=1"

        # Test HTTPS URL with subdomain
        model = DummyModel(url="https://sub.example.com")
        assert model.url == "https://sub.example.com"

        # Test HTTPS URL with port
        model = DummyModel(url="https://example.com:8443")
        assert model.url == "https://example.com:8443"

        # Test HTTPS URL with whitespace
        model = DummyModel(url="  https://example.com  ")
        assert model.url == "https://example.com"

        # TODO: should we even allow this?
        # Test HTTPS URL with port
        model = DummyModel(url="https://example.com:8443")
        assert model.url == "https://example.com:8443"

    def test_valid_data_uris(self):
        # Test basic data URI
        model = DummyModel(url="data:text/plain;base64,SGVsbG8gV29ybGQ=")
        assert model.url == "data:text/plain;base64,SGVsbG8gV29ybGQ="

        # Test data URI with image
        image_uri = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="  # noqa: E501
        model = DummyModel(url=image_uri)
        assert model.url == image_uri

        # Test data URI with whitespace
        model = DummyModel(url="  data:text/plain,Hello World  ")
        assert model.url == "data:text/plain,Hello World"

    def test_invalid_inputs(self):
        # Test HTTP URL (non-HTTPS)
        with pytest.raises(ValueError):
            DummyModel(url="http://example.com")

        # Test malformed URL
        with pytest.raises(ValueError):
            DummyModel(url="not-a-url")

        # Test invalid data URI
        with pytest.raises(ValueError):
            DummyModel(url="invalid-data-uri")

        # Test empty string
        with pytest.raises(ValueError):
            DummyModel(url="")

        # Test None value
        with pytest.raises(ValueError):
            DummyModel(url=None)

    def test_length_limits(self):
        # Test HTTPS URL at max length
        domain = "example.com"
        path_length = MAX_HTTPS_URL_LENGTH - len(f"https://{domain}/")
        long_url = f"https://{domain}/{'a' * path_length}"
        model = DummyModel(url=long_url)
        assert model.url == long_url

        # Test HTTPS URL exceeding max length
        too_long_url = f"https://example.com/{'a' * MAX_HTTPS_URL_LENGTH}"
        with pytest.raises(ValidationError):
            DummyModel(url=too_long_url)

        # Test data URI at max length
        uri_prefix = "data:text/plain,"
        long_uri = f"{uri_prefix}{'a' * (MAX_DATA_URI_LENGTH - len(uri_prefix))}"
        model = DummyModel(url=long_uri)
        assert model.url == long_uri

        # Test data URI exceeding max length
        too_long_uri = f"data:text/plain,{'a' * MAX_DATA_URI_LENGTH}"
        with pytest.raises(ValueError):
            DummyModel(url=too_long_uri)

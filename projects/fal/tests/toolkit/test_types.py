from collections.abc import Generator
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image as PILImage
from pydantic import BaseModel, Field, ValidationError

from fal.toolkit.exceptions import (
    FileDownloadException,
    FileTooLargeException,
    ImageAspectRatioException,
    ImageLoadException,
    ImageTooLargeException,
    ImageTooSmallException,
    ToolkitDataFormatException,
    ToolkitFileDownloadException,
    ToolkitFileSizeExceededException,
    ToolkitImageLoadException,
)
from fal.toolkit.types import (
    AudioUrl,
    FieldContextProviderModel,
    HttpsOrDataUrl,
    ImageMaskUrl,
    ImageUrl,
    VideoUrl,
    ZipUrl,
)


class DummyModel(FieldContextProviderModel):
    image_url: ImageUrl


def create_image_url(url_string: str) -> ImageUrl:
    """Helper function to create an ImageUrl instance within a model context."""
    model = DummyModel(image_url=url_string)
    return model.image_url


@pytest.fixture
def image_url() -> ImageUrl:
    """Provides a basic, valid ImageUrl instance."""
    return create_image_url("https://example.com/photo.png")


def test_comma_in_image_url():
    return create_image_url("https://example.com/photo,1.png")


def test_query_params_in_image_url():
    return create_image_url("https://example.com/photo.png?q=1")


def test_trailing_garbage_in_image_url():
    with pytest.raises(ValidationError):
        create_image_url("https://example.com/image.jpg garbage")


@pytest.fixture
def mock_pil_image() -> PILImage.Image:
    """Provides a standard mock PIL Image object (ratio 1.5)."""
    return PILImage.new("RGB", (300, 200), color="blue")


@pytest.fixture
def mock_read_image_sync(
    mock_pil_image: PILImage.Image,
) -> Generator[MagicMock, None, None]:
    """Patches the synchronous image reading function."""
    with patch(
        "fal.toolkit.types.read_image_from_url", return_value=mock_pil_image
    ) as mock:
        yield mock


@pytest.fixture
def mock_read_image_async(
    mock_pil_image: PILImage.Image,
) -> Generator[MagicMock, None, None]:
    """Patches the asynchronous image reading function."""
    with patch(
        "fal.toolkit.types.read_image_from_url_async", return_value=mock_pil_image
    ) as mock:
        yield mock


class TestFieldContextProvider:
    """Tests the FieldContextProviderModel for correct context binding."""

    def test_nested_model_context_binding(self):
        class NestedModel(FieldContextProviderModel):
            photo: ImageUrl

        class ParentModel(FieldContextProviderModel):
            nested: NestedModel

        model = ParentModel.parse_obj(
            {"nested": {"photo": "https://example.com/photo.png"}}
        )
        assert model.nested.photo._loc == ("nested", "photo")

    def test_list_of_urls_context_binding(self):
        class ListModel(FieldContextProviderModel):
            images: list[ImageUrl]

        model = ListModel.parse_obj(
            {
                "images": [
                    "https://api.aa.international/1.jpg",
                    "https://bb.com/2.jpg",
                    "https://v3.fal.media/files/penguin/qmLZSvOIzTKs6bDFXiEtH_video.mp4",
                ]
            }
        )
        assert model.images[0]._loc == ("images", 0)

    def test_deeply_nested_model_context_binding(self):
        class SubItem(FieldContextProviderModel):
            photo: ImageUrl

        class ItemModel(FieldContextProviderModel):
            sub_item: SubItem

        class ParentListModel(FieldContextProviderModel):
            items: list[ItemModel]

        data = {"items": [{"sub_item": {"photo": "https://example.com/item1.png"}}]}
        model = ParentListModel.parse_obj(data)
        assert model.items[0].sub_item.photo._loc == ("items", 0, "sub_item", "photo")

    def test_deeply_nested_model_exception_loc_field(
        self, mock_read_image_sync: MagicMock
    ):
        """Tests that exceptions from nested ImageUrl fields contain correct loc
        information."""

        class SubItem(FieldContextProviderModel):
            photo: ImageUrl

        class ItemModel(FieldContextProviderModel):
            sub_item: SubItem

        class ParentListModel(FieldContextProviderModel):
            items: list[ItemModel]

        data = {
            "items": [
                {"sub_item": {"photo": "https://example.com/item1.png"}},
                {"sub_item": {"photo": "https://example.com/item2.png"}},
            ]
        }
        # Create a model with deeply nested ImageUrl
        model = ParentListModel.parse_obj(data)

        # Mock the image reading to return an image that will fail validation
        mock_read_image_sync.return_value = PILImage.new("RGB", (50, 50))

        # Call to_pil with parameters that will cause validation to fail
        with pytest.raises(ImageTooSmallException) as exc_info:
            model.items[0].sub_item.photo.to_pil(min_width=100)

        # Verify that the exception contains the correct loc information
        assert exc_info.value.detail[0]["loc"] == (
            "body",
            "items",
            0,
            "sub_item",
            "photo",
        )  # type: ignore


class TestImageUrlPydanticValidation:
    """Tests for initial validation at the Pydantic level."""

    @pytest.mark.parametrize(
        "url",
        [
            "http://example.com/image.jpg",
            "https://example.com/image.jpg",
            "https://example.com:8080/image.jpg",
            "https://sub.domain.com:65535/path/to/image.png",
            "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=",
        ],
    )
    def test_valid_urls(self, url):
        assert str(create_image_url(url)) == url

    def test_invalid_input_type(self):
        with pytest.raises(ValidationError):
            create_image_url(123)  # type: ignore

    @pytest.mark.parametrize(
        "invalid_url",
        [
            "https//example.com",
            "https: example.com",
            "ftp://example.com/image.jpg",
            "data:not-a-valid-uri",
            # outside of valid port range
            "http://example.com:0/image.jpg",
            "https://example.com:65536/image.jpg",
        ],
    )
    def test_invalid_url_formats(self, invalid_url: str):
        """Tests that subtly malformed URLs fail validation."""
        with pytest.raises(ValidationError):
            create_image_url(invalid_url)


class TestImageUrlSchema:
    """Tests for the __modify_schema__ method."""

    def test_json_schema_generation(self):
        ui_config = {"important": True}
        extras = {"a": "b", "c": 1}

        class ModelWithImage(FieldContextProviderModel):
            avatar: ImageUrl = Field(
                description="User avatar", ui=ui_config, dummy=extras
            )

        schema = ModelWithImage.schema()
        field_schema = schema["properties"]["avatar"]

        for k, v in ui_config.items():
            assert field_schema["ui"][k] == v

        for k, v in extras.items():
            assert field_schema["dummy"][k] == v

        assert field_schema["description"] == "User avatar"
        assert field_schema["type"] == "string"


class TestUiFieldNameDefaults:
    @pytest.mark.parametrize(
        "UrlType,expected",
        [
            (HttpsOrDataUrl, "url"),
            (ImageUrl, "image"),
            (ImageMaskUrl, "image_mask"),
            (VideoUrl, "video"),
            (AudioUrl, "audio"),
            (ZipUrl, "archive"),
        ],
    )
    def test_ui_field_default_per_type(self, UrlType, expected):
        class Plain(BaseModel):
            asset: UrlType = Field(description="Asset")

        schema = Plain.schema()
        field_schema = schema["properties"]["asset"]
        assert field_schema["ui"]["field"] == expected

    def test_ui_field_preserves_override(self):
        class Plain(BaseModel):
            asset: ImageUrl = Field(
                description="Asset", ui={"field": "custom", "extra": 1}
            )

        schema = Plain.schema()
        field_schema = schema["properties"]["asset"]
        assert field_schema["ui"]["field"] == "custom"
        assert field_schema["ui"]["extra"] == 1


class TestImageUrlLoading:
    """Tests for downloading/loading the image and validating its properties."""


class TestExtrasPropagationOnPlainModel:
    def test_plain_model_field_extras_propagate_image_url(self):
        ui_config = {"important": True}
        extras = {"a": "b", "c": 1}

        class Plain(BaseModel):
            asset: ImageUrl | None = Field(
                description="Asset", ui=ui_config, dummy=extras, default=None
            )

        schema = Plain.schema()
        field_schema = schema["properties"]["asset"]

        for k, v in ui_config.items():
            assert field_schema["ui"][k] == v

        for k, v in extras.items():
            assert field_schema["dummy"][k] == v

        assert field_schema.get("field") is None
        assert field_schema["ui"]["field"] == "image"

    def test_plain_model_field_extras_propagate_https_or_data_url(self):
        ui_config = {"flag": 1}
        meta = {"x": True, "y": 2}

        class Plain(BaseModel):
            url: HttpsOrDataUrl = Field(description="Url", ui=ui_config, meta=meta)

        schema = Plain.schema()
        field_schema = schema["properties"]["url"]

        for k, v in ui_config.items():
            assert field_schema["ui"][k] == v

        for k, v in meta.items():
            assert field_schema["meta"][k] == v

        assert field_schema["ui"]["field"] == "url"

    def test_to_pil_success(
        self,
        image_url: ImageUrl,
        mock_read_image_sync: MagicMock,
        mock_pil_image: PILImage.Image,
    ):
        result = image_url.to_pil()
        assert result == mock_pil_image
        mock_read_image_sync.assert_called_once_with(
            image_url, timeout=20.0, max_size=None
        )

    @pytest.mark.parametrize("width, height", [(100, 0), (0, 100)])
    def test_load_fails_for_zero_or_negative_dimension_image(
        self,
        image_url: ImageUrl,
        mock_read_image_sync: MagicMock,
        width: int,
        height: int,
    ):
        """Tests that loading an image with a non-positive dimension fails early."""
        if width == 0 or height == 0:
            img = PILImage.new("RGB", (100, 100))
            img = img.crop((0, 0, width, height))
            mock_read_image_sync.return_value = img
        else:
            mock_read_image_sync.return_value = PILImage.new("RGB", (width, height))

        with pytest.raises(ImageLoadException):
            image_url.to_pil()

    @pytest.mark.parametrize(
        "image_size, min_ratio, max_ratio, should_pass",
        [
            ((10, 400), None, None, True),
            ((10, 400), 1.0, None, False),
            ((10, 400), None, 1.0, False),
            ((400, 400), 1.0, 1.0, True),
            ((10, 400), 10 / 400 + 1, 400 / 10, False),
            ((10, 400), 10 / 400, 400 / 10 - 1, False),
            ((10, 400), 10 / 400, 400 / 10, True),
            ((400, 10), 10 / 400 + 1, 400 / 10, False),
            ((400, 10), 10 / 400, 400 / 10 - 1, False),
            ((400, 10), 10 / 400, 400 / 10, True),
        ],
    )
    def test_aspect_ratio_validation(
        self,
        image_url: ImageUrl,
        mock_read_image_sync: MagicMock,
        image_size,
        min_ratio,
        max_ratio,
        should_pass,
    ):
        """Tests the corrected aspect ratio logic."""
        mock_read_image_sync.return_value = PILImage.new("RGB", image_size)
        options = {"min_aspect_ratio": min_ratio, "max_aspect_ratio": max_ratio}
        if should_pass:
            image_url.to_pil(**options)
        else:
            if min_ratio is not None and max_ratio is not None:
                # both values are provided
                with pytest.raises(ImageAspectRatioException):
                    image_url.to_pil(**options)
            else:
                # one of the values is not provided
                with pytest.raises(ValueError):
                    image_url.to_pil(**options)

    def test_error_message_contains_correct_details(
        self, image_url, mock_read_image_sync
    ):
        """Tests that raised exceptions contain the correct validation values."""
        mock_read_image_sync.return_value = PILImage.new("RGB", (500, 500))
        with pytest.raises(ImageTooLargeException) as exc_info:
            image_url.to_pil(max_width=400, max_height=450)

        assert exc_info.value.detail[0]["ctx"]["max_height"] == 450  # type: ignore
        assert exc_info.value.detail[0]["ctx"]["max_width"] == 400  # type: ignore

        mock_read_image_sync.side_effect = ToolkitFileSizeExceededException("too big")
        with pytest.raises(FileTooLargeException) as exc_info:
            image_url.to_pil(max_file_size=1024)
        assert exc_info.value.detail[0]["ctx"]["max_size"] == 1024  # type: ignore


class TestImageUrlErrorHandling:
    """Tests for correct mapping of Toolkit to public exceptions."""

    @pytest.mark.parametrize(
        "Toolkit_error, public_error",
        [
            (ToolkitImageLoadException, ImageLoadException),
            (ToolkitFileDownloadException, FileDownloadException),
            (ToolkitFileSizeExceededException, FileTooLargeException),
            (ToolkitDataFormatException, ImageLoadException),
        ],
    )
    def test_error_mapping(
        self,
        image_url: ImageUrl,
        mock_read_image_sync: MagicMock,
        Toolkit_error,
        public_error,
    ):
        mock_read_image_sync.side_effect = Toolkit_error("mock error")
        with pytest.raises(public_error):
            image_url.to_pil()

    def test_unhandled_exception_is_propagated(
        self, image_url: ImageUrl, mock_read_image_sync: MagicMock
    ):
        mock_read_image_sync.side_effect = ValueError("A generic, unexpected error")
        with pytest.raises(ValueError, match="A generic, unexpected error"):
            image_url.to_pil()


class TestDerivedUrlTypes:
    """Tests that derived URL classes behave exactly like their parents."""

    def test_image_mask_url_is_an_image_url(self, mock_read_image_sync: MagicMock):
        """ImageMaskUrl should inherit all of ImageUrl's validation logic."""

        class Model(FieldContextProviderModel):
            mask: ImageMaskUrl

        instance = Model(mask="https://example.com/mask.png")
        assert isinstance(instance.mask, ImageUrl)

        mock_read_image_sync.return_value = PILImage.new("RGB", (50, 50))
        with pytest.raises(ImageTooSmallException):
            instance.mask.to_pil(min_width=100)

    @pytest.mark.parametrize("UrlClass", [ZipUrl, VideoUrl, AudioUrl])
    def test_other_urls_are_https_or_data_urls(self, UrlClass):
        """ZipUrl, VideoUrl, and AudioUrl should have no custom validation logic."""

        class Model(FieldContextProviderModel):
            asset: UrlClass

        # Should accept a valid HTTPS URL, even with a "wrong" extension
        valid_url = "https://example.com/some_file.txt"
        instance = Model(asset=valid_url)
        assert str(instance.asset) == valid_url

        # Should accept a valid data URI
        data_uri = "data:application/octet-stream;base64,SGVsbG8h"
        instance = Model(asset=data_uri)
        assert str(instance.asset) == data_uri

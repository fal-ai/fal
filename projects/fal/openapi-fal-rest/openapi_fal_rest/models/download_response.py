from typing import Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

T = TypeVar("T", bound="DownloadResponse")


@_attrs_define
class DownloadResponse:
    """
    Attributes:
        download_urls (list[str]):
        expires_in_seconds (int):
    """

    download_urls: list[str]
    expires_in_seconds: int
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        download_urls = self.download_urls

        expires_in_seconds = self.expires_in_seconds

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "download_urls": download_urls,
                "expires_in_seconds": expires_in_seconds,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        download_urls = cast(list[str], d.pop("download_urls"))

        expires_in_seconds = d.pop("expires_in_seconds")

        download_response = cls(
            download_urls=download_urls,
            expires_in_seconds=expires_in_seconds,
        )

        download_response.additional_properties = d
        return download_response

    @property
    def additional_keys(self) -> list[str]:
        return list(self.additional_properties.keys())

    def __getitem__(self, key: str) -> Any:
        return self.additional_properties[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.additional_properties[key] = value

    def __delitem__(self, key: str) -> None:
        del self.additional_properties[key]

    def __contains__(self, key: str) -> bool:
        return key in self.additional_properties

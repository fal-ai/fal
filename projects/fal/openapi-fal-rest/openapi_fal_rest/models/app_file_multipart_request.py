from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

T = TypeVar("T", bound="AppFileMultipartRequest")


@_attrs_define
class AppFileMultipartRequest:
    """
    Attributes:
        hash_ (str):
        mode (str):
        mtime (str):
        size (str):
    """

    hash_: str
    mode: str
    mtime: str
    size: str
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        hash_ = self.hash_

        mode = self.mode

        mtime = self.mtime

        size = self.size

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "hash": hash_,
                "mode": mode,
                "mtime": mtime,
                "size": size,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        hash_ = d.pop("hash")

        mode = d.pop("mode")

        mtime = d.pop("mtime")

        size = d.pop("size")

        app_file_multipart_request = cls(
            hash_=hash_,
            mode=mode,
            mtime=mtime,
            size=size,
        )

        app_file_multipart_request.additional_properties = d
        return app_file_multipart_request

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

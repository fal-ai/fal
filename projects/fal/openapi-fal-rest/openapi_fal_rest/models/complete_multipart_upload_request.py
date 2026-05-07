from typing import TYPE_CHECKING, Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.app_file_multipart_request import AppFileMultipartRequest
    from ..models.upload_multipart_part_response import UploadMultipartPartResponse


T = TypeVar("T", bound="CompleteMultipartUploadRequest")


@_attrs_define
class CompleteMultipartUploadRequest:
    """
    Attributes:
        parts (list['UploadMultipartPartResponse']):
        metadata (Union[Unset, AppFileMultipartRequest]):
    """

    parts: list["UploadMultipartPartResponse"]
    metadata: Union[Unset, "AppFileMultipartRequest"] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        parts = []
        for parts_item_data in self.parts:
            parts_item = parts_item_data.to_dict()
            parts.append(parts_item)

        metadata: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.metadata, Unset):
            metadata = self.metadata.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "parts": parts,
            }
        )
        if metadata is not UNSET:
            field_dict["metadata"] = metadata

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        from ..models.app_file_multipart_request import AppFileMultipartRequest
        from ..models.upload_multipart_part_response import UploadMultipartPartResponse

        d = src_dict.copy()
        parts = []
        _parts = d.pop("parts")
        for parts_item_data in _parts:
            parts_item = UploadMultipartPartResponse.from_dict(parts_item_data)

            parts.append(parts_item)

        _metadata = d.pop("metadata", UNSET)
        metadata: Union[Unset, AppFileMultipartRequest]
        if isinstance(_metadata, Unset):
            metadata = UNSET
        else:
            metadata = AppFileMultipartRequest.from_dict(_metadata)

        complete_multipart_upload_request = cls(
            parts=parts,
            metadata=metadata,
        )

        complete_multipart_upload_request.additional_properties = d
        return complete_multipart_upload_request

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

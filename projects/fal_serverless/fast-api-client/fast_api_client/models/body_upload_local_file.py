from io import BytesIO
from typing import Any, Dict, List, Type, TypeVar

import attr

from ..types import File

T = TypeVar("T", bound="BodyUploadLocalFile")


@attr.s(auto_attribs=True)
class BodyUploadLocalFile:
    """
    Attributes:
        file_upload (File):
    """

    file_upload: File
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        file_upload = self.file_upload.to_tuple()

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "file_upload": file_upload,
            }
        )

        return field_dict

    def to_multipart(self) -> Dict[str, Any]:
        file_upload = self.file_upload.to_tuple()

        field_dict: Dict[str, Any] = {}
        field_dict.update(
            {key: (None, str(value).encode(), "text/plain") for key, value in self.additional_properties.items()}
        )
        field_dict.update(
            {
                "file_upload": file_upload,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        file_upload = File(payload=BytesIO(d.pop("file_upload")))

        body_upload_local_file = cls(
            file_upload=file_upload,
        )

        body_upload_local_file.additional_properties = d
        return body_upload_local_file

    @property
    def additional_keys(self) -> List[str]:
        return list(self.additional_properties.keys())

    def __getitem__(self, key: str) -> Any:
        return self.additional_properties[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.additional_properties[key] = value

    def __delitem__(self, key: str) -> None:
        del self.additional_properties[key]

    def __contains__(self, key: str) -> bool:
        return key in self.additional_properties

from typing import Any, Dict, List, Type, TypeVar

import attr

T = TypeVar("T", bound="InitiateUploadInfo")


@attr.s(auto_attribs=True)
class InitiateUploadInfo:
    """
    Attributes:
        file_name (str):
        content_type (str):
    """

    file_name: str
    content_type: str
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        file_name = self.file_name
        content_type = self.content_type

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "file_name": file_name,
                "content_type": content_type,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        file_name = d.pop("file_name")

        content_type = d.pop("content_type")

        initiate_upload_info = cls(
            file_name=file_name,
            content_type=content_type,
        )

        initiate_upload_info.additional_properties = d
        return initiate_upload_info

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

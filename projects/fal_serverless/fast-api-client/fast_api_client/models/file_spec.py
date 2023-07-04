from typing import Any, Dict, List, Type, TypeVar, Union

import attr

from ..types import UNSET, Unset

T = TypeVar("T", bound="FileSpec")


@attr.s(auto_attribs=True)
class FileSpec:
    """
    Attributes:
        path (str):
        name (str):
        created_time (float):
        updated_time (float):
        is_file (bool):
        size (int):
        checksum_sha256 (Union[Unset, str]):
        checksum_md5 (Union[Unset, str]):
    """

    path: str
    name: str
    created_time: float
    updated_time: float
    is_file: bool
    size: int
    checksum_sha256: Union[Unset, str] = UNSET
    checksum_md5: Union[Unset, str] = UNSET
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        path = self.path
        name = self.name
        created_time = self.created_time
        updated_time = self.updated_time
        is_file = self.is_file
        size = self.size
        checksum_sha256 = self.checksum_sha256
        checksum_md5 = self.checksum_md5

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "path": path,
                "name": name,
                "created_time": created_time,
                "updated_time": updated_time,
                "is_file": is_file,
                "size": size,
            }
        )
        if checksum_sha256 is not UNSET:
            field_dict["checksum_sha256"] = checksum_sha256
        if checksum_md5 is not UNSET:
            field_dict["checksum_md5"] = checksum_md5

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        path = d.pop("path")

        name = d.pop("name")

        created_time = d.pop("created_time")

        updated_time = d.pop("updated_time")

        is_file = d.pop("is_file")

        size = d.pop("size")

        checksum_sha256 = d.pop("checksum_sha256", UNSET)

        checksum_md5 = d.pop("checksum_md5", UNSET)

        file_spec = cls(
            path=path,
            name=name,
            created_time=created_time,
            updated_time=updated_time,
            is_file=is_file,
            size=size,
            checksum_sha256=checksum_sha256,
            checksum_md5=checksum_md5,
        )

        file_spec.additional_properties = d
        return file_spec

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

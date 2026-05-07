from typing import Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="NamespaceIndexInfo")


@_attrs_define
class NamespaceIndexInfo:
    """
    Attributes:
        status (str):
        unindexed_bytes (Union[Unset, int]):
    """

    status: str
    unindexed_bytes: Union[Unset, int] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        status = self.status

        unindexed_bytes = self.unindexed_bytes

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "status": status,
            }
        )
        if unindexed_bytes is not UNSET:
            field_dict["unindexed_bytes"] = unindexed_bytes

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        status = d.pop("status")

        unindexed_bytes = d.pop("unindexed_bytes", UNSET)

        namespace_index_info = cls(
            status=status,
            unindexed_bytes=unindexed_bytes,
        )

        namespace_index_info.additional_properties = d
        return namespace_index_info

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

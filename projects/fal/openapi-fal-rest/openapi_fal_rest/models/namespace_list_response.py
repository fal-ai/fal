from typing import TYPE_CHECKING, Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.namespace_info import NamespaceInfo


T = TypeVar("T", bound="NamespaceListResponse")


@_attrs_define
class NamespaceListResponse:
    """
    Attributes:
        namespaces (list['NamespaceInfo']):
        next_cursor (Union[Unset, str]):
    """

    namespaces: list["NamespaceInfo"]
    next_cursor: Union[Unset, str] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        namespaces = []
        for namespaces_item_data in self.namespaces:
            namespaces_item = namespaces_item_data.to_dict()
            namespaces.append(namespaces_item)

        next_cursor = self.next_cursor

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "namespaces": namespaces,
            }
        )
        if next_cursor is not UNSET:
            field_dict["next_cursor"] = next_cursor

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        from ..models.namespace_info import NamespaceInfo

        d = src_dict.copy()
        namespaces = []
        _namespaces = d.pop("namespaces")
        for namespaces_item_data in _namespaces:
            namespaces_item = NamespaceInfo.from_dict(namespaces_item_data)

            namespaces.append(namespaces_item)

        next_cursor = d.pop("next_cursor", UNSET)

        namespace_list_response = cls(
            namespaces=namespaces,
            next_cursor=next_cursor,
        )

        namespace_list_response.additional_properties = d
        return namespace_list_response

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

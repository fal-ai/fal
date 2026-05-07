from typing import Any, TypeVar, Union, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="DocumentQualifyResponse")


@_attrs_define
class DocumentQualifyResponse:
    """
    Attributes:
        eligible (bool):
        item_types (Union[Unset, list[str]]):
        item_count (Union[Unset, int]):  Default: 0.
        reason (Union[Unset, str]):
    """

    eligible: bool
    item_types: Union[Unset, list[str]] = UNSET
    item_count: Union[Unset, int] = 0
    reason: Union[Unset, str] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        eligible = self.eligible

        item_types: Union[Unset, list[str]] = UNSET
        if not isinstance(self.item_types, Unset):
            item_types = self.item_types

        item_count = self.item_count

        reason = self.reason

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "eligible": eligible,
            }
        )
        if item_types is not UNSET:
            field_dict["item_types"] = item_types
        if item_count is not UNSET:
            field_dict["item_count"] = item_count
        if reason is not UNSET:
            field_dict["reason"] = reason

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        eligible = d.pop("eligible")

        item_types = cast(list[str], d.pop("item_types", UNSET))

        item_count = d.pop("item_count", UNSET)

        reason = d.pop("reason", UNSET)

        document_qualify_response = cls(
            eligible=eligible,
            item_types=item_types,
            item_count=item_count,
            reason=reason,
        )

        document_qualify_response.additional_properties = d
        return document_qualify_response

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

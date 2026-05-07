from typing import Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="MarkAllReadResponse")


@_attrs_define
class MarkAllReadResponse:
    """
    Attributes:
        marked_count (int):
        has_more (Union[Unset, bool]):  Default: False.
    """

    marked_count: int
    has_more: Union[Unset, bool] = False
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        marked_count = self.marked_count

        has_more = self.has_more

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "marked_count": marked_count,
            }
        )
        if has_more is not UNSET:
            field_dict["has_more"] = has_more

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        marked_count = d.pop("marked_count")

        has_more = d.pop("has_more", UNSET)

        mark_all_read_response = cls(
            marked_count=marked_count,
            has_more=has_more,
        )

        mark_all_read_response.additional_properties = d
        return mark_all_read_response

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

from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

T = TypeVar("T", bound="CountStats")


@_attrs_define
class CountStats:
    """
    Attributes:
        request_count (int):
        success_count (int):
        error_count (int):
        user_error_count (int):
    """

    request_count: int
    success_count: int
    error_count: int
    user_error_count: int
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        request_count = self.request_count

        success_count = self.success_count

        error_count = self.error_count

        user_error_count = self.user_error_count

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "request_count": request_count,
                "success_count": success_count,
                "error_count": error_count,
                "user_error_count": user_error_count,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        request_count = d.pop("request_count")

        success_count = d.pop("success_count")

        error_count = d.pop("error_count")

        user_error_count = d.pop("user_error_count")

        count_stats = cls(
            request_count=request_count,
            success_count=success_count,
            error_count=error_count,
            user_error_count=user_error_count,
        )

        count_stats.additional_properties = d
        return count_stats

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

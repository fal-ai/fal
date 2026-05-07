from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

T = TypeVar("T", bound="BillingReportEntry")


@_attrs_define
class BillingReportEntry:
    """
    Attributes:
        week_name (str):
        week_start (str):
        week_end (str):
        size_bytes (int):
        created_at (str):
    """

    week_name: str
    week_start: str
    week_end: str
    size_bytes: int
    created_at: str
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        week_name = self.week_name

        week_start = self.week_start

        week_end = self.week_end

        size_bytes = self.size_bytes

        created_at = self.created_at

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "week_name": week_name,
                "week_start": week_start,
                "week_end": week_end,
                "size_bytes": size_bytes,
                "created_at": created_at,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        week_name = d.pop("week_name")

        week_start = d.pop("week_start")

        week_end = d.pop("week_end")

        size_bytes = d.pop("size_bytes")

        created_at = d.pop("created_at")

        billing_report_entry = cls(
            week_name=week_name,
            week_start=week_start,
            week_end=week_end,
            size_bytes=size_bytes,
            created_at=created_at,
        )

        billing_report_entry.additional_properties = d
        return billing_report_entry

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

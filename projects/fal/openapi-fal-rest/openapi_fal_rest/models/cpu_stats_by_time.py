import datetime
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from dateutil.parser import isoparse

T = TypeVar("T", bound="CpuStatsByTime")


@_attrs_define
class CpuStatsByTime:
    """
    Attributes:
        datetime_ (datetime.datetime):
        cpu_usage_percent (float):
    """

    datetime_: datetime.datetime
    cpu_usage_percent: float
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        datetime_ = self.datetime_.isoformat()

        cpu_usage_percent = self.cpu_usage_percent

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "datetime": datetime_,
                "cpu_usage_percent": cpu_usage_percent,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        datetime_ = isoparse(d.pop("datetime"))

        cpu_usage_percent = d.pop("cpu_usage_percent")

        cpu_stats_by_time = cls(
            datetime_=datetime_,
            cpu_usage_percent=cpu_usage_percent,
        )

        cpu_stats_by_time.additional_properties = d
        return cpu_stats_by_time

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

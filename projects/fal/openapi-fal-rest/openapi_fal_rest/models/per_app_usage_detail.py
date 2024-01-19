import datetime
from typing import Any, Dict, List, Type, TypeVar

import attr
from dateutil.parser import isoparse

T = TypeVar("T", bound="PerAppUsageDetail")


@attr.s(auto_attribs=True)
class PerAppUsageDetail:
    """
    Attributes:
        date (datetime.datetime):
        machine_type (str):
        request_count (int):
        application_alias (str):
        total_billable_duration (int):
    """

    date: datetime.datetime
    machine_type: str
    request_count: int
    application_alias: str
    total_billable_duration: int
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        date = self.date.isoformat()

        machine_type = self.machine_type
        request_count = self.request_count
        application_alias = self.application_alias
        total_billable_duration = self.total_billable_duration

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "date": date,
                "machine_type": machine_type,
                "request_count": request_count,
                "application_alias": application_alias,
                "total_billable_duration": total_billable_duration,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        date = isoparse(d.pop("date"))

        machine_type = d.pop("machine_type")

        request_count = d.pop("request_count")

        application_alias = d.pop("application_alias")

        total_billable_duration = d.pop("total_billable_duration")

        per_app_usage_detail = cls(
            date=date,
            machine_type=machine_type,
            request_count=request_count,
            application_alias=application_alias,
            total_billable_duration=total_billable_duration,
        )

        per_app_usage_detail.additional_properties = d
        return per_app_usage_detail

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

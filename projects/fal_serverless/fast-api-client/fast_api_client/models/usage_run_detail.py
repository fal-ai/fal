import datetime
from typing import Any, Dict, List, Type, TypeVar

import attr
from dateutil.parser import isoparse

T = TypeVar("T", bound="UsageRunDetail")


@attr.s(auto_attribs=True)
class UsageRunDetail:
    """
    Attributes:
        id (str):
        duration_seconds (int):
        start_time (datetime.date):
    """

    id: str
    duration_seconds: int
    start_time: datetime.date
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        id = self.id
        duration_seconds = self.duration_seconds
        start_time = self.start_time.isoformat()

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "id": id,
                "duration_seconds": duration_seconds,
                "start_time": start_time,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        id = d.pop("id")

        duration_seconds = d.pop("duration_seconds")

        start_time = isoparse(d.pop("start_time")).date()

        usage_run_detail = cls(
            id=id,
            duration_seconds=duration_seconds,
            start_time=start_time,
        )

        usage_run_detail.additional_properties = d
        return usage_run_detail

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

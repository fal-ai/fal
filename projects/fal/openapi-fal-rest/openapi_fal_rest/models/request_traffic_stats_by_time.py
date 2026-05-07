import datetime
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from dateutil.parser import isoparse

T = TypeVar("T", bound="RequestTrafficStatsByTime")


@_attrs_define
class RequestTrafficStatsByTime:
    """
    Attributes:
        datetime_ (datetime.datetime):
        received_per_second (float):
        processed_per_second (float):
        concurrent_requests (float):
    """

    datetime_: datetime.datetime
    received_per_second: float
    processed_per_second: float
    concurrent_requests: float
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        datetime_ = self.datetime_.isoformat()

        received_per_second = self.received_per_second

        processed_per_second = self.processed_per_second

        concurrent_requests = self.concurrent_requests

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "datetime": datetime_,
                "received_per_second": received_per_second,
                "processed_per_second": processed_per_second,
                "concurrent_requests": concurrent_requests,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        datetime_ = isoparse(d.pop("datetime"))

        received_per_second = d.pop("received_per_second")

        processed_per_second = d.pop("processed_per_second")

        concurrent_requests = d.pop("concurrent_requests")

        request_traffic_stats_by_time = cls(
            datetime_=datetime_,
            received_per_second=received_per_second,
            processed_per_second=processed_per_second,
            concurrent_requests=concurrent_requests,
        )

        request_traffic_stats_by_time.additional_properties = d
        return request_traffic_stats_by_time

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

import datetime
from typing import Any, TypeVar, Union, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from dateutil.parser import isoparse

T = TypeVar("T", bound="RequestLatencyStatsByTime")


@_attrs_define
class RequestLatencyStatsByTime:
    """
    Attributes:
        datetime_ (datetime.datetime):
        p50 (Union[None, float]):
        p90 (Union[None, float]):
        p95 (Union[None, float]):
        p99 (Union[None, float]):
    """

    datetime_: datetime.datetime
    p50: Union[None, float]
    p90: Union[None, float]
    p95: Union[None, float]
    p99: Union[None, float]
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        datetime_ = self.datetime_.isoformat()

        p50: Union[None, float]
        p50 = self.p50

        p90: Union[None, float]
        p90 = self.p90

        p95: Union[None, float]
        p95 = self.p95

        p99: Union[None, float]
        p99 = self.p99

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "datetime": datetime_,
                "p50": p50,
                "p90": p90,
                "p95": p95,
                "p99": p99,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        datetime_ = isoparse(d.pop("datetime"))

        def _parse_p50(data: object) -> Union[None, float]:
            if data is None:
                return data
            return cast(Union[None, float], data)

        p50 = _parse_p50(d.pop("p50"))

        def _parse_p90(data: object) -> Union[None, float]:
            if data is None:
                return data
            return cast(Union[None, float], data)

        p90 = _parse_p90(d.pop("p90"))

        def _parse_p95(data: object) -> Union[None, float]:
            if data is None:
                return data
            return cast(Union[None, float], data)

        p95 = _parse_p95(d.pop("p95"))

        def _parse_p99(data: object) -> Union[None, float]:
            if data is None:
                return data
            return cast(Union[None, float], data)

        p99 = _parse_p99(d.pop("p99"))

        request_latency_stats_by_time = cls(
            datetime_=datetime_,
            p50=p50,
            p90=p90,
            p95=p95,
            p99=p99,
        )

        request_latency_stats_by_time.additional_properties = d
        return request_latency_stats_by_time

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

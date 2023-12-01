import datetime
from typing import Any, Dict, List, Type, TypeVar, Union

import attr
from dateutil.parser import isoparse

from ..types import UNSET, Unset

T = TypeVar("T", bound="GatewayStatsByTime")


@attr.s(auto_attribs=True)
class GatewayStatsByTime:
    """
    Attributes:
        request_count (int):
        success_count (int):
        error_count (int):
        datetime_ (datetime.datetime):
        p25_duration (Union[Unset, None, float]):
        p50_duration (Union[Unset, None, float]):
        p75_duration (Union[Unset, None, float]):
        p90_duration (Union[Unset, None, float]):
    """

    request_count: int
    success_count: int
    error_count: int
    datetime_: datetime.datetime
    p25_duration: Union[Unset, None, float] = UNSET
    p50_duration: Union[Unset, None, float] = UNSET
    p75_duration: Union[Unset, None, float] = UNSET
    p90_duration: Union[Unset, None, float] = UNSET
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        request_count = self.request_count
        success_count = self.success_count
        error_count = self.error_count
        datetime_ = self.datetime_.isoformat()

        p25_duration = self.p25_duration
        p50_duration = self.p50_duration
        p75_duration = self.p75_duration
        p90_duration = self.p90_duration

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "request_count": request_count,
                "success_count": success_count,
                "error_count": error_count,
                "datetime": datetime_,
            }
        )
        if p25_duration is not UNSET:
            field_dict["p25_duration"] = p25_duration
        if p50_duration is not UNSET:
            field_dict["p50_duration"] = p50_duration
        if p75_duration is not UNSET:
            field_dict["p75_duration"] = p75_duration
        if p90_duration is not UNSET:
            field_dict["p90_duration"] = p90_duration

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        request_count = d.pop("request_count")

        success_count = d.pop("success_count")

        error_count = d.pop("error_count")

        datetime_ = isoparse(d.pop("datetime"))

        p25_duration = d.pop("p25_duration", UNSET)

        p50_duration = d.pop("p50_duration", UNSET)

        p75_duration = d.pop("p75_duration", UNSET)

        p90_duration = d.pop("p90_duration", UNSET)

        gateway_stats_by_time = cls(
            request_count=request_count,
            success_count=success_count,
            error_count=error_count,
            datetime_=datetime_,
            p25_duration=p25_duration,
            p50_duration=p50_duration,
            p75_duration=p75_duration,
            p90_duration=p90_duration,
        )

        gateway_stats_by_time.additional_properties = d
        return gateway_stats_by_time

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

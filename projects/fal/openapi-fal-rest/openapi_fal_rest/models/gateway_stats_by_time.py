import datetime
from typing import Any, TypeVar, Union, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from dateutil.parser import isoparse

T = TypeVar("T", bound="GatewayStatsByTime")


@_attrs_define
class GatewayStatsByTime:
    """
    Attributes:
        request_count (int):
        success_count (int):
        error_count (int):
        user_error_count (int):
        p25_duration (Union[None, float]):
        p50_duration (Union[None, float]):
        p75_duration (Union[None, float]):
        p90_duration (Union[None, float]):
        p25_prepare_duration (Union[None, float]):
        p50_prepare_duration (Union[None, float]):
        p75_prepare_duration (Union[None, float]):
        p90_prepare_duration (Union[None, float]):
        datetime_ (datetime.datetime):
    """

    request_count: int
    success_count: int
    error_count: int
    user_error_count: int
    p25_duration: Union[None, float]
    p50_duration: Union[None, float]
    p75_duration: Union[None, float]
    p90_duration: Union[None, float]
    p25_prepare_duration: Union[None, float]
    p50_prepare_duration: Union[None, float]
    p75_prepare_duration: Union[None, float]
    p90_prepare_duration: Union[None, float]
    datetime_: datetime.datetime
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        request_count = self.request_count

        success_count = self.success_count

        error_count = self.error_count

        user_error_count = self.user_error_count

        p25_duration: Union[None, float]
        p25_duration = self.p25_duration

        p50_duration: Union[None, float]
        p50_duration = self.p50_duration

        p75_duration: Union[None, float]
        p75_duration = self.p75_duration

        p90_duration: Union[None, float]
        p90_duration = self.p90_duration

        p25_prepare_duration: Union[None, float]
        p25_prepare_duration = self.p25_prepare_duration

        p50_prepare_duration: Union[None, float]
        p50_prepare_duration = self.p50_prepare_duration

        p75_prepare_duration: Union[None, float]
        p75_prepare_duration = self.p75_prepare_duration

        p90_prepare_duration: Union[None, float]
        p90_prepare_duration = self.p90_prepare_duration

        datetime_ = self.datetime_.isoformat()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "request_count": request_count,
                "success_count": success_count,
                "error_count": error_count,
                "user_error_count": user_error_count,
                "p25_duration": p25_duration,
                "p50_duration": p50_duration,
                "p75_duration": p75_duration,
                "p90_duration": p90_duration,
                "p25_prepare_duration": p25_prepare_duration,
                "p50_prepare_duration": p50_prepare_duration,
                "p75_prepare_duration": p75_prepare_duration,
                "p90_prepare_duration": p90_prepare_duration,
                "datetime": datetime_,
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

        def _parse_p25_duration(data: object) -> Union[None, float]:
            if data is None:
                return data
            return cast(Union[None, float], data)

        p25_duration = _parse_p25_duration(d.pop("p25_duration"))

        def _parse_p50_duration(data: object) -> Union[None, float]:
            if data is None:
                return data
            return cast(Union[None, float], data)

        p50_duration = _parse_p50_duration(d.pop("p50_duration"))

        def _parse_p75_duration(data: object) -> Union[None, float]:
            if data is None:
                return data
            return cast(Union[None, float], data)

        p75_duration = _parse_p75_duration(d.pop("p75_duration"))

        def _parse_p90_duration(data: object) -> Union[None, float]:
            if data is None:
                return data
            return cast(Union[None, float], data)

        p90_duration = _parse_p90_duration(d.pop("p90_duration"))

        def _parse_p25_prepare_duration(data: object) -> Union[None, float]:
            if data is None:
                return data
            return cast(Union[None, float], data)

        p25_prepare_duration = _parse_p25_prepare_duration(d.pop("p25_prepare_duration"))

        def _parse_p50_prepare_duration(data: object) -> Union[None, float]:
            if data is None:
                return data
            return cast(Union[None, float], data)

        p50_prepare_duration = _parse_p50_prepare_duration(d.pop("p50_prepare_duration"))

        def _parse_p75_prepare_duration(data: object) -> Union[None, float]:
            if data is None:
                return data
            return cast(Union[None, float], data)

        p75_prepare_duration = _parse_p75_prepare_duration(d.pop("p75_prepare_duration"))

        def _parse_p90_prepare_duration(data: object) -> Union[None, float]:
            if data is None:
                return data
            return cast(Union[None, float], data)

        p90_prepare_duration = _parse_p90_prepare_duration(d.pop("p90_prepare_duration"))

        datetime_ = isoparse(d.pop("datetime"))

        gateway_stats_by_time = cls(
            request_count=request_count,
            success_count=success_count,
            error_count=error_count,
            user_error_count=user_error_count,
            p25_duration=p25_duration,
            p50_duration=p50_duration,
            p75_duration=p75_duration,
            p90_duration=p90_duration,
            p25_prepare_duration=p25_prepare_duration,
            p50_prepare_duration=p50_prepare_duration,
            p75_prepare_duration=p75_prepare_duration,
            p90_prepare_duration=p90_prepare_duration,
            datetime_=datetime_,
        )

        gateway_stats_by_time.additional_properties = d
        return gateway_stats_by_time

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

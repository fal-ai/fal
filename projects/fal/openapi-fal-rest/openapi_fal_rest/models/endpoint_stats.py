from typing import TYPE_CHECKING, Any, TypeVar, Union, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

if TYPE_CHECKING:
    from ..models.time_stats import TimeStats


T = TypeVar("T", bound="EndpointStats")


@_attrs_define
class EndpointStats:
    """
    Attributes:
        endpoint (str):
        request_count (int):
        error_count (int):
        success_count (int):
        user_error_count (int):
        p25_duration (Union[None, float]):
        p50_duration (Union[None, float]):
        p75_duration (Union[None, float]):
        p90_duration (Union[None, float]):
        time_stats (list['TimeStats']):
    """

    endpoint: str
    request_count: int
    error_count: int
    success_count: int
    user_error_count: int
    p25_duration: Union[None, float]
    p50_duration: Union[None, float]
    p75_duration: Union[None, float]
    p90_duration: Union[None, float]
    time_stats: list["TimeStats"]
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        endpoint = self.endpoint

        request_count = self.request_count

        error_count = self.error_count

        success_count = self.success_count

        user_error_count = self.user_error_count

        p25_duration: Union[None, float]
        p25_duration = self.p25_duration

        p50_duration: Union[None, float]
        p50_duration = self.p50_duration

        p75_duration: Union[None, float]
        p75_duration = self.p75_duration

        p90_duration: Union[None, float]
        p90_duration = self.p90_duration

        time_stats = []
        for time_stats_item_data in self.time_stats:
            time_stats_item = time_stats_item_data.to_dict()
            time_stats.append(time_stats_item)

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "endpoint": endpoint,
                "request_count": request_count,
                "error_count": error_count,
                "success_count": success_count,
                "user_error_count": user_error_count,
                "p25_duration": p25_duration,
                "p50_duration": p50_duration,
                "p75_duration": p75_duration,
                "p90_duration": p90_duration,
                "time_stats": time_stats,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        from ..models.time_stats import TimeStats

        d = src_dict.copy()
        endpoint = d.pop("endpoint")

        request_count = d.pop("request_count")

        error_count = d.pop("error_count")

        success_count = d.pop("success_count")

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

        time_stats = []
        _time_stats = d.pop("time_stats")
        for time_stats_item_data in _time_stats:
            time_stats_item = TimeStats.from_dict(time_stats_item_data)

            time_stats.append(time_stats_item)

        endpoint_stats = cls(
            endpoint=endpoint,
            request_count=request_count,
            error_count=error_count,
            success_count=success_count,
            user_error_count=user_error_count,
            p25_duration=p25_duration,
            p50_duration=p50_duration,
            p75_duration=p75_duration,
            p90_duration=p90_duration,
            time_stats=time_stats,
        )

        endpoint_stats.additional_properties = d
        return endpoint_stats

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

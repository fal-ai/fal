from typing import TYPE_CHECKING, Any, Dict, List, Type, TypeVar, Union

import attr

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.gateway_stats_by_time import GatewayStatsByTime


T = TypeVar("T", bound="GatewayUsageStats")


@attr.s(auto_attribs=True)
class GatewayUsageStats:
    """
    Attributes:
        request_count (int):
        success_count (int):
        error_count (int):
        application_name (str):
        total_billable_duration (float):
        p25_duration (Union[Unset, None, float]):
        p50_duration (Union[Unset, None, float]):
        p75_duration (Union[Unset, None, float]):
        p90_duration (Union[Unset, None, float]):
        time_stats (Union[Unset, None, List['GatewayStatsByTime']]):
    """

    request_count: int
    success_count: int
    error_count: int
    application_name: str
    total_billable_duration: float
    p25_duration: Union[Unset, None, float] = UNSET
    p50_duration: Union[Unset, None, float] = UNSET
    p75_duration: Union[Unset, None, float] = UNSET
    p90_duration: Union[Unset, None, float] = UNSET
    time_stats: Union[Unset, None, List["GatewayStatsByTime"]] = UNSET
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        request_count = self.request_count
        success_count = self.success_count
        error_count = self.error_count
        application_name = self.application_name
        total_billable_duration = self.total_billable_duration
        p25_duration = self.p25_duration
        p50_duration = self.p50_duration
        p75_duration = self.p75_duration
        p90_duration = self.p90_duration
        time_stats: Union[Unset, None, List[Dict[str, Any]]] = UNSET
        if not isinstance(self.time_stats, Unset):
            if self.time_stats is None:
                time_stats = None
            else:
                time_stats = []
                for time_stats_item_data in self.time_stats:
                    time_stats_item = time_stats_item_data.to_dict()

                    time_stats.append(time_stats_item)

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "request_count": request_count,
                "success_count": success_count,
                "error_count": error_count,
                "application_name": application_name,
                "total_billable_duration": total_billable_duration,
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
        if time_stats is not UNSET:
            field_dict["time_stats"] = time_stats

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        from ..models.gateway_stats_by_time import GatewayStatsByTime

        d = src_dict.copy()
        request_count = d.pop("request_count")

        success_count = d.pop("success_count")

        error_count = d.pop("error_count")

        application_name = d.pop("application_name")

        total_billable_duration = d.pop("total_billable_duration")

        p25_duration = d.pop("p25_duration", UNSET)

        p50_duration = d.pop("p50_duration", UNSET)

        p75_duration = d.pop("p75_duration", UNSET)

        p90_duration = d.pop("p90_duration", UNSET)

        time_stats = []
        _time_stats = d.pop("time_stats", UNSET)
        for time_stats_item_data in _time_stats or []:
            time_stats_item = GatewayStatsByTime.from_dict(time_stats_item_data)

            time_stats.append(time_stats_item)

        gateway_usage_stats = cls(
            request_count=request_count,
            success_count=success_count,
            error_count=error_count,
            application_name=application_name,
            total_billable_duration=total_billable_duration,
            p25_duration=p25_duration,
            p50_duration=p50_duration,
            p75_duration=p75_duration,
            p90_duration=p90_duration,
            time_stats=time_stats,
        )

        gateway_usage_stats.additional_properties = d
        return gateway_usage_stats

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

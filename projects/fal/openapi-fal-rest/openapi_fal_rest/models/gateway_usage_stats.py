from typing import TYPE_CHECKING, Any, TypeVar, Union, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.gateway_stats_by_time import GatewayStatsByTime


T = TypeVar("T", bound="GatewayUsageStats")


@_attrs_define
class GatewayUsageStats:
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
        application_name (str):
        application_user_id (str):
        total_billable_duration (float):
        machine_type (Union[None, str]):
        total_duration (Union[None, float]):
        time_stats (Union[None, list['GatewayStatsByTime']]):
        endpoint (Union[None, Unset, str]):
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
    application_name: str
    application_user_id: str
    total_billable_duration: float
    machine_type: Union[None, str]
    total_duration: Union[None, float]
    time_stats: Union[None, list["GatewayStatsByTime"]]
    endpoint: Union[None, Unset, str] = UNSET
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

        application_name = self.application_name

        application_user_id = self.application_user_id

        total_billable_duration = self.total_billable_duration

        machine_type: Union[None, str]
        machine_type = self.machine_type

        total_duration: Union[None, float]
        total_duration = self.total_duration

        time_stats: Union[None, list[dict[str, Any]]]
        if isinstance(self.time_stats, list):
            time_stats = []
            for time_stats_type_0_item_data in self.time_stats:
                time_stats_type_0_item = time_stats_type_0_item_data.to_dict()
                time_stats.append(time_stats_type_0_item)

        else:
            time_stats = self.time_stats

        endpoint: Union[None, Unset, str]
        if isinstance(self.endpoint, Unset):
            endpoint = UNSET
        else:
            endpoint = self.endpoint

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
                "application_name": application_name,
                "application_user_id": application_user_id,
                "total_billable_duration": total_billable_duration,
                "machine_type": machine_type,
                "total_duration": total_duration,
                "time_stats": time_stats,
            }
        )
        if endpoint is not UNSET:
            field_dict["endpoint"] = endpoint

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        from ..models.gateway_stats_by_time import GatewayStatsByTime

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

        application_name = d.pop("application_name")

        application_user_id = d.pop("application_user_id")

        total_billable_duration = d.pop("total_billable_duration")

        def _parse_machine_type(data: object) -> Union[None, str]:
            if data is None:
                return data
            return cast(Union[None, str], data)

        machine_type = _parse_machine_type(d.pop("machine_type"))

        def _parse_total_duration(data: object) -> Union[None, float]:
            if data is None:
                return data
            return cast(Union[None, float], data)

        total_duration = _parse_total_duration(d.pop("total_duration"))

        def _parse_time_stats(data: object) -> Union[None, list["GatewayStatsByTime"]]:
            if data is None:
                return data
            try:
                if not isinstance(data, list):
                    raise TypeError()
                time_stats_type_0 = []
                _time_stats_type_0 = data
                for time_stats_type_0_item_data in _time_stats_type_0:
                    time_stats_type_0_item = GatewayStatsByTime.from_dict(time_stats_type_0_item_data)

                    time_stats_type_0.append(time_stats_type_0_item)

                return time_stats_type_0
            except:  # noqa: E722
                pass
            return cast(Union[None, list["GatewayStatsByTime"]], data)

        time_stats = _parse_time_stats(d.pop("time_stats"))

        def _parse_endpoint(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        endpoint = _parse_endpoint(d.pop("endpoint", UNSET))

        gateway_usage_stats = cls(
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
            application_name=application_name,
            application_user_id=application_user_id,
            total_billable_duration=total_billable_duration,
            machine_type=machine_type,
            total_duration=total_duration,
            time_stats=time_stats,
            endpoint=endpoint,
        )

        gateway_usage_stats.additional_properties = d
        return gateway_usage_stats

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

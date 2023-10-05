from typing import Any, Dict, List, Type, TypeVar

import attr

T = TypeVar("T", bound="GatewayUsageStats")


@attr.s(auto_attribs=True)
class GatewayUsageStats:
    """
    Attributes:
        application_id (str):
        application_alias (str):
        request_count (int):
        success_count (int):
        error_count (int):
        total_billable_duration (int):
        p25_duration (float):
        p50_duration (float):
        p75_duration (float):
    """

    application_id: str
    application_alias: str
    request_count: int
    success_count: int
    error_count: int
    total_billable_duration: int
    p25_duration: float
    p50_duration: float
    p75_duration: float
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        application_id = self.application_id
        application_alias = self.application_alias
        request_count = self.request_count
        success_count = self.success_count
        error_count = self.error_count
        total_billable_duration = self.total_billable_duration
        p25_duration = self.p25_duration
        p50_duration = self.p50_duration
        p75_duration = self.p75_duration

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "application_id": application_id,
                "application_alias": application_alias,
                "request_count": request_count,
                "success_count": success_count,
                "error_count": error_count,
                "total_billable_duration": total_billable_duration,
                "p25_duration": p25_duration,
                "p50_duration": p50_duration,
                "p75_duration": p75_duration,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        application_id = d.pop("application_id")

        application_alias = d.pop("application_alias")

        request_count = d.pop("request_count")

        success_count = d.pop("success_count")

        error_count = d.pop("error_count")

        total_billable_duration = d.pop("total_billable_duration")

        p25_duration = d.pop("p25_duration")

        p50_duration = d.pop("p50_duration")

        p75_duration = d.pop("p75_duration")

        gateway_usage_stats = cls(
            application_id=application_id,
            application_alias=application_alias,
            request_count=request_count,
            success_count=success_count,
            error_count=error_count,
            total_billable_duration=total_billable_duration,
            p25_duration=p25_duration,
            p50_duration=p50_duration,
            p75_duration=p75_duration,
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

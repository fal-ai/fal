from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.gateway_request_traffic_stats_request_type import GatewayRequestTrafficStatsRequestType

if TYPE_CHECKING:
    from ..models.request_traffic_stats_by_time import RequestTrafficStatsByTime


T = TypeVar("T", bound="GatewayRequestTrafficStats")


@_attrs_define
class GatewayRequestTrafficStats:
    """
    Attributes:
        endpoint (str):
        application_name (str):
        application_user_id (str):
        request_type (GatewayRequestTrafficStatsRequestType):
        time_stats (list['RequestTrafficStatsByTime']):
    """

    endpoint: str
    application_name: str
    application_user_id: str
    request_type: GatewayRequestTrafficStatsRequestType
    time_stats: list["RequestTrafficStatsByTime"]
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        endpoint = self.endpoint

        application_name = self.application_name

        application_user_id = self.application_user_id

        request_type = self.request_type.value

        time_stats = []
        for time_stats_item_data in self.time_stats:
            time_stats_item = time_stats_item_data.to_dict()
            time_stats.append(time_stats_item)

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "endpoint": endpoint,
                "application_name": application_name,
                "application_user_id": application_user_id,
                "request_type": request_type,
                "time_stats": time_stats,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        from ..models.request_traffic_stats_by_time import RequestTrafficStatsByTime

        d = src_dict.copy()
        endpoint = d.pop("endpoint")

        application_name = d.pop("application_name")

        application_user_id = d.pop("application_user_id")

        request_type = GatewayRequestTrafficStatsRequestType(d.pop("request_type"))

        time_stats = []
        _time_stats = d.pop("time_stats")
        for time_stats_item_data in _time_stats:
            time_stats_item = RequestTrafficStatsByTime.from_dict(time_stats_item_data)

            time_stats.append(time_stats_item)

        gateway_request_traffic_stats = cls(
            endpoint=endpoint,
            application_name=application_name,
            application_user_id=application_user_id,
            request_type=request_type,
            time_stats=time_stats,
        )

        gateway_request_traffic_stats.additional_properties = d
        return gateway_request_traffic_stats

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

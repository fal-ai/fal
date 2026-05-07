from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.requests_by_status_code_stats_request_type import RequestsByStatusCodeStatsRequestType

if TYPE_CHECKING:
    from ..models.requests_by_status_code_point import RequestsByStatusCodePoint


T = TypeVar("T", bound="RequestsByStatusCodeStats")


@_attrs_define
class RequestsByStatusCodeStats:
    """
    Attributes:
        request_type (RequestsByStatusCodeStatsRequestType):
        time_stats (list['RequestsByStatusCodePoint']):
    """

    request_type: RequestsByStatusCodeStatsRequestType
    time_stats: list["RequestsByStatusCodePoint"]
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        request_type = self.request_type.value

        time_stats = []
        for time_stats_item_data in self.time_stats:
            time_stats_item = time_stats_item_data.to_dict()
            time_stats.append(time_stats_item)

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "request_type": request_type,
                "time_stats": time_stats,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        from ..models.requests_by_status_code_point import RequestsByStatusCodePoint

        d = src_dict.copy()
        request_type = RequestsByStatusCodeStatsRequestType(d.pop("request_type"))

        time_stats = []
        _time_stats = d.pop("time_stats")
        for time_stats_item_data in _time_stats:
            time_stats_item = RequestsByStatusCodePoint.from_dict(time_stats_item_data)

            time_stats.append(time_stats_item)

        requests_by_status_code_stats = cls(
            request_type=request_type,
            time_stats=time_stats,
        )

        requests_by_status_code_stats.additional_properties = d
        return requests_by_status_code_stats

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

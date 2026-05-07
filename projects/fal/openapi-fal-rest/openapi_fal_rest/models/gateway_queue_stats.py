from typing import TYPE_CHECKING, Any, TypeVar, Union, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

if TYPE_CHECKING:
    from ..models.queue_stats_by_time import QueueStatsByTime


T = TypeVar("T", bound="GatewayQueueStats")


@_attrs_define
class GatewayQueueStats:
    """
    Attributes:
        size (int):
        endpoint (str):
        application_name (str):
        application_user_id (str):
        time_stats (Union[None, list['QueueStatsByTime']]):
    """

    size: int
    endpoint: str
    application_name: str
    application_user_id: str
    time_stats: Union[None, list["QueueStatsByTime"]]
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        size = self.size

        endpoint = self.endpoint

        application_name = self.application_name

        application_user_id = self.application_user_id

        time_stats: Union[None, list[dict[str, Any]]]
        if isinstance(self.time_stats, list):
            time_stats = []
            for time_stats_type_0_item_data in self.time_stats:
                time_stats_type_0_item = time_stats_type_0_item_data.to_dict()
                time_stats.append(time_stats_type_0_item)

        else:
            time_stats = self.time_stats

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "size": size,
                "endpoint": endpoint,
                "application_name": application_name,
                "application_user_id": application_user_id,
                "time_stats": time_stats,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        from ..models.queue_stats_by_time import QueueStatsByTime

        d = src_dict.copy()
        size = d.pop("size")

        endpoint = d.pop("endpoint")

        application_name = d.pop("application_name")

        application_user_id = d.pop("application_user_id")

        def _parse_time_stats(data: object) -> Union[None, list["QueueStatsByTime"]]:
            if data is None:
                return data
            try:
                if not isinstance(data, list):
                    raise TypeError()
                time_stats_type_0 = []
                _time_stats_type_0 = data
                for time_stats_type_0_item_data in _time_stats_type_0:
                    time_stats_type_0_item = QueueStatsByTime.from_dict(time_stats_type_0_item_data)

                    time_stats_type_0.append(time_stats_type_0_item)

                return time_stats_type_0
            except:  # noqa: E722
                pass
            return cast(Union[None, list["QueueStatsByTime"]], data)

        time_stats = _parse_time_stats(d.pop("time_stats"))

        gateway_queue_stats = cls(
            size=size,
            endpoint=endpoint,
            application_name=application_name,
            application_user_id=application_user_id,
            time_stats=time_stats,
        )

        gateway_queue_stats.additional_properties = d
        return gateway_queue_stats

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

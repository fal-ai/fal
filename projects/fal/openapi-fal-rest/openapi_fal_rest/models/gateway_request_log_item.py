from typing import TYPE_CHECKING, Any, TypeVar
from uuid import UUID

from attrs import define as _attrs_define
from attrs import field as _attrs_field

if TYPE_CHECKING:
    from ..models.gateway_request_log_item_logs_item import GatewayRequestLogItemLogsItem


T = TypeVar("T", bound="GatewayRequestLogItem")


@_attrs_define
class GatewayRequestLogItem:
    """
    Attributes:
        request_id (UUID):
        logs (list['GatewayRequestLogItemLogsItem']):
    """

    request_id: UUID
    logs: list["GatewayRequestLogItemLogsItem"]
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        request_id = str(self.request_id)

        logs = []
        for logs_item_data in self.logs:
            logs_item = logs_item_data.to_dict()
            logs.append(logs_item)

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "request_id": request_id,
                "logs": logs,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        from ..models.gateway_request_log_item_logs_item import GatewayRequestLogItemLogsItem

        d = src_dict.copy()
        request_id = UUID(d.pop("request_id"))

        logs = []
        _logs = d.pop("logs")
        for logs_item_data in _logs:
            logs_item = GatewayRequestLogItemLogsItem.from_dict(logs_item_data)

            logs.append(logs_item)

        gateway_request_log_item = cls(
            request_id=request_id,
            logs=logs,
        )

        gateway_request_log_item.additional_properties = d
        return gateway_request_log_item

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

from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

if TYPE_CHECKING:
    from ..models.alertmanager_webhook_alert import AlertmanagerWebhookAlert


T = TypeVar("T", bound="AlertmanagerWebhookPayload")


@_attrs_define
class AlertmanagerWebhookPayload:
    """
    Attributes:
        alerts (list['AlertmanagerWebhookAlert']):
    """

    alerts: list["AlertmanagerWebhookAlert"]
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        alerts = []
        for alerts_item_data in self.alerts:
            alerts_item = alerts_item_data.to_dict()
            alerts.append(alerts_item)

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "alerts": alerts,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        from ..models.alertmanager_webhook_alert import AlertmanagerWebhookAlert

        d = src_dict.copy()
        alerts = []
        _alerts = d.pop("alerts")
        for alerts_item_data in _alerts:
            alerts_item = AlertmanagerWebhookAlert.from_dict(alerts_item_data)

            alerts.append(alerts_item)

        alertmanager_webhook_payload = cls(
            alerts=alerts,
        )

        alertmanager_webhook_payload.additional_properties = d
        return alertmanager_webhook_payload

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

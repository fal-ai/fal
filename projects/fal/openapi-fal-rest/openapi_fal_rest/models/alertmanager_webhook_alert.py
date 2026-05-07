from typing import TYPE_CHECKING, Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.alertmanager_webhook_alert_annotations import AlertmanagerWebhookAlertAnnotations
    from ..models.alertmanager_webhook_alert_labels import AlertmanagerWebhookAlertLabels


T = TypeVar("T", bound="AlertmanagerWebhookAlert")


@_attrs_define
class AlertmanagerWebhookAlert:
    """
    Attributes:
        status (str):
        labels (AlertmanagerWebhookAlertLabels):
        annotations (Union[Unset, AlertmanagerWebhookAlertAnnotations]):
    """

    status: str
    labels: "AlertmanagerWebhookAlertLabels"
    annotations: Union[Unset, "AlertmanagerWebhookAlertAnnotations"] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        status = self.status

        labels = self.labels.to_dict()

        annotations: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.annotations, Unset):
            annotations = self.annotations.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "status": status,
                "labels": labels,
            }
        )
        if annotations is not UNSET:
            field_dict["annotations"] = annotations

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        from ..models.alertmanager_webhook_alert_annotations import AlertmanagerWebhookAlertAnnotations
        from ..models.alertmanager_webhook_alert_labels import AlertmanagerWebhookAlertLabels

        d = src_dict.copy()
        status = d.pop("status")

        labels = AlertmanagerWebhookAlertLabels.from_dict(d.pop("labels"))

        _annotations = d.pop("annotations", UNSET)
        annotations: Union[Unset, AlertmanagerWebhookAlertAnnotations]
        if isinstance(_annotations, Unset):
            annotations = UNSET
        else:
            annotations = AlertmanagerWebhookAlertAnnotations.from_dict(_annotations)

        alertmanager_webhook_alert = cls(
            status=status,
            labels=labels,
            annotations=annotations,
        )

        alertmanager_webhook_alert.additional_properties = d
        return alertmanager_webhook_alert

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

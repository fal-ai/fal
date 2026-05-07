import datetime
from typing import TYPE_CHECKING, Any, Literal, TypeVar, Union, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from dateutil.parser import isoparse

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.deployment_event_payload import DeploymentEventPayload


T = TypeVar("T", bound="DeploymentRollingFailedEvent")


@_attrs_define
class DeploymentRollingFailedEvent:
    """
    Attributes:
        event_id (str):
        user_id (str):
        application_id (str):
        created_at (datetime.datetime):
        category (Literal['deployment_rolling_failed']):
        payload (DeploymentEventPayload):
        app_name (Union[Unset, str]):
    """

    event_id: str
    user_id: str
    application_id: str
    created_at: datetime.datetime
    category: Literal["deployment_rolling_failed"]
    payload: "DeploymentEventPayload"
    app_name: Union[Unset, str] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        event_id = self.event_id

        user_id = self.user_id

        application_id = self.application_id

        created_at = self.created_at.isoformat()

        category = self.category

        payload = self.payload.to_dict()

        app_name = self.app_name

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "event_id": event_id,
                "user_id": user_id,
                "application_id": application_id,
                "created_at": created_at,
                "category": category,
                "payload": payload,
            }
        )
        if app_name is not UNSET:
            field_dict["app_name"] = app_name

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        from ..models.deployment_event_payload import DeploymentEventPayload

        d = src_dict.copy()
        event_id = d.pop("event_id")

        user_id = d.pop("user_id")

        application_id = d.pop("application_id")

        created_at = isoparse(d.pop("created_at"))

        category = cast(Literal["deployment_rolling_failed"], d.pop("category"))
        if category != "deployment_rolling_failed":
            raise ValueError(f"category must match const 'deployment_rolling_failed', got '{category}'")

        payload = DeploymentEventPayload.from_dict(d.pop("payload"))

        app_name = d.pop("app_name", UNSET)

        deployment_rolling_failed_event = cls(
            event_id=event_id,
            user_id=user_id,
            application_id=application_id,
            created_at=created_at,
            category=category,
            payload=payload,
            app_name=app_name,
        )

        deployment_rolling_failed_event.additional_properties = d
        return deployment_rolling_failed_event

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

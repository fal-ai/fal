from typing import TYPE_CHECKING, Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.event_actor import EventActor


T = TypeVar("T", bound="DeploymentEventPayload")


@_attrs_define
class DeploymentEventPayload:
    """
    Attributes:
        new_application_id (str):
        old_application_id (Union[Unset, str]):
        actor (Union[Unset, EventActor]):
    """

    new_application_id: str
    old_application_id: Union[Unset, str] = UNSET
    actor: Union[Unset, "EventActor"] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        new_application_id = self.new_application_id

        old_application_id = self.old_application_id

        actor: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.actor, Unset):
            actor = self.actor.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "new_application_id": new_application_id,
            }
        )
        if old_application_id is not UNSET:
            field_dict["old_application_id"] = old_application_id
        if actor is not UNSET:
            field_dict["actor"] = actor

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        from ..models.event_actor import EventActor

        d = src_dict.copy()
        new_application_id = d.pop("new_application_id")

        old_application_id = d.pop("old_application_id", UNSET)

        _actor = d.pop("actor", UNSET)
        actor: Union[Unset, EventActor]
        if isinstance(_actor, Unset):
            actor = UNSET
        else:
            actor = EventActor.from_dict(_actor)

        deployment_event_payload = cls(
            new_application_id=new_application_id,
            old_application_id=old_application_id,
            actor=actor,
        )

        deployment_event_payload.additional_properties = d
        return deployment_event_payload

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

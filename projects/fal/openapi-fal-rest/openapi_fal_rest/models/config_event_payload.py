from typing import TYPE_CHECKING, Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.config_event_payload_new_config import ConfigEventPayloadNewConfig
    from ..models.config_event_payload_old_config import ConfigEventPayloadOldConfig
    from ..models.event_actor import EventActor


T = TypeVar("T", bound="ConfigEventPayload")


@_attrs_define
class ConfigEventPayload:
    """
    Attributes:
        old_config (Union[Unset, ConfigEventPayloadOldConfig]):
        new_config (Union[Unset, ConfigEventPayloadNewConfig]):
        old_app_auth_mode (Union[Unset, str]):
        new_app_auth_mode (Union[Unset, str]):
        actor (Union[Unset, EventActor]):
    """

    old_config: Union[Unset, "ConfigEventPayloadOldConfig"] = UNSET
    new_config: Union[Unset, "ConfigEventPayloadNewConfig"] = UNSET
    old_app_auth_mode: Union[Unset, str] = UNSET
    new_app_auth_mode: Union[Unset, str] = UNSET
    actor: Union[Unset, "EventActor"] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        old_config: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.old_config, Unset):
            old_config = self.old_config.to_dict()

        new_config: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.new_config, Unset):
            new_config = self.new_config.to_dict()

        old_app_auth_mode = self.old_app_auth_mode

        new_app_auth_mode = self.new_app_auth_mode

        actor: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.actor, Unset):
            actor = self.actor.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if old_config is not UNSET:
            field_dict["old_config"] = old_config
        if new_config is not UNSET:
            field_dict["new_config"] = new_config
        if old_app_auth_mode is not UNSET:
            field_dict["old_app_auth_mode"] = old_app_auth_mode
        if new_app_auth_mode is not UNSET:
            field_dict["new_app_auth_mode"] = new_app_auth_mode
        if actor is not UNSET:
            field_dict["actor"] = actor

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        from ..models.config_event_payload_new_config import ConfigEventPayloadNewConfig
        from ..models.config_event_payload_old_config import ConfigEventPayloadOldConfig
        from ..models.event_actor import EventActor

        d = src_dict.copy()
        _old_config = d.pop("old_config", UNSET)
        old_config: Union[Unset, ConfigEventPayloadOldConfig]
        if isinstance(_old_config, Unset):
            old_config = UNSET
        else:
            old_config = ConfigEventPayloadOldConfig.from_dict(_old_config)

        _new_config = d.pop("new_config", UNSET)
        new_config: Union[Unset, ConfigEventPayloadNewConfig]
        if isinstance(_new_config, Unset):
            new_config = UNSET
        else:
            new_config = ConfigEventPayloadNewConfig.from_dict(_new_config)

        old_app_auth_mode = d.pop("old_app_auth_mode", UNSET)

        new_app_auth_mode = d.pop("new_app_auth_mode", UNSET)

        _actor = d.pop("actor", UNSET)
        actor: Union[Unset, EventActor]
        if isinstance(_actor, Unset):
            actor = UNSET
        else:
            actor = EventActor.from_dict(_actor)

        config_event_payload = cls(
            old_config=old_config,
            new_config=new_config,
            old_app_auth_mode=old_app_auth_mode,
            new_app_auth_mode=new_app_auth_mode,
            actor=actor,
        )

        config_event_payload.additional_properties = d
        return config_event_payload

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

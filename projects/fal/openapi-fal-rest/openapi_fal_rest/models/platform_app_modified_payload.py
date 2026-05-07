from typing import TYPE_CHECKING, Any, TypeVar, Union, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.platform_app_modified_actor import PlatformAppModifiedActor
    from ..models.platform_app_modified_payload_config import PlatformAppModifiedPayloadConfig


T = TypeVar("T", bound="PlatformAppModifiedPayload")


@_attrs_define
class PlatformAppModifiedPayload:
    """
    Attributes:
        message (Union[Unset, str]):
        application (Union[Unset, str]):
        full_application (Union[Unset, str]):
        environment (Union[Unset, str]):
        auth_method (Union[Unset, str]):
        app_auth_mode (Union[Unset, str]):
        old_app_auth_mode (Union[Unset, str]):
        config (Union['PlatformAppModifiedPayloadConfig', None, Unset]):
        app_update_type (Union[Unset, str]):
        actor (Union[Unset, PlatformAppModifiedActor]):
    """

    message: Union[Unset, str] = UNSET
    application: Union[Unset, str] = UNSET
    full_application: Union[Unset, str] = UNSET
    environment: Union[Unset, str] = UNSET
    auth_method: Union[Unset, str] = UNSET
    app_auth_mode: Union[Unset, str] = UNSET
    old_app_auth_mode: Union[Unset, str] = UNSET
    config: Union["PlatformAppModifiedPayloadConfig", None, Unset] = UNSET
    app_update_type: Union[Unset, str] = UNSET
    actor: Union[Unset, "PlatformAppModifiedActor"] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        from ..models.platform_app_modified_payload_config import PlatformAppModifiedPayloadConfig

        message = self.message

        application = self.application

        full_application = self.full_application

        environment = self.environment

        auth_method = self.auth_method

        app_auth_mode = self.app_auth_mode

        old_app_auth_mode = self.old_app_auth_mode

        config: Union[None, Unset, dict[str, Any]]
        if isinstance(self.config, Unset):
            config = UNSET
        elif isinstance(self.config, PlatformAppModifiedPayloadConfig):
            config = self.config.to_dict()
        else:
            config = self.config

        app_update_type = self.app_update_type

        actor: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.actor, Unset):
            actor = self.actor.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if message is not UNSET:
            field_dict["message"] = message
        if application is not UNSET:
            field_dict["application"] = application
        if full_application is not UNSET:
            field_dict["full_application"] = full_application
        if environment is not UNSET:
            field_dict["environment"] = environment
        if auth_method is not UNSET:
            field_dict["auth_method"] = auth_method
        if app_auth_mode is not UNSET:
            field_dict["app_auth_mode"] = app_auth_mode
        if old_app_auth_mode is not UNSET:
            field_dict["old_app_auth_mode"] = old_app_auth_mode
        if config is not UNSET:
            field_dict["config"] = config
        if app_update_type is not UNSET:
            field_dict["app_update_type"] = app_update_type
        if actor is not UNSET:
            field_dict["actor"] = actor

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        from ..models.platform_app_modified_actor import PlatformAppModifiedActor
        from ..models.platform_app_modified_payload_config import PlatformAppModifiedPayloadConfig

        d = src_dict.copy()
        message = d.pop("message", UNSET)

        application = d.pop("application", UNSET)

        full_application = d.pop("full_application", UNSET)

        environment = d.pop("environment", UNSET)

        auth_method = d.pop("auth_method", UNSET)

        app_auth_mode = d.pop("app_auth_mode", UNSET)

        old_app_auth_mode = d.pop("old_app_auth_mode", UNSET)

        def _parse_config(data: object) -> Union["PlatformAppModifiedPayloadConfig", None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                config_type_0 = PlatformAppModifiedPayloadConfig.from_dict(data)

                return config_type_0
            except:  # noqa: E722
                pass
            return cast(Union["PlatformAppModifiedPayloadConfig", None, Unset], data)

        config = _parse_config(d.pop("config", UNSET))

        app_update_type = d.pop("app_update_type", UNSET)

        _actor = d.pop("actor", UNSET)
        actor: Union[Unset, PlatformAppModifiedActor]
        if isinstance(_actor, Unset):
            actor = UNSET
        else:
            actor = PlatformAppModifiedActor.from_dict(_actor)

        platform_app_modified_payload = cls(
            message=message,
            application=application,
            full_application=full_application,
            environment=environment,
            auth_method=auth_method,
            app_auth_mode=app_auth_mode,
            old_app_auth_mode=old_app_auth_mode,
            config=config,
            app_update_type=app_update_type,
            actor=actor,
        )

        platform_app_modified_payload.additional_properties = d
        return platform_app_modified_payload

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

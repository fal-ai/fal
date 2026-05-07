import datetime
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from dateutil.parser import isoparse

if TYPE_CHECKING:
    from ..models.sso_connection_settings_response_settings import SSOConnectionSettingsResponseSettings


T = TypeVar("T", bound="SSOConnectionSettingsResponse")


@_attrs_define
class SSOConnectionSettingsResponse:
    """
    Attributes:
        sso_connection (str):
        settings (SSOConnectionSettingsResponseSettings):
        created_at (datetime.datetime):
        updated_at (datetime.datetime):
    """

    sso_connection: str
    settings: "SSOConnectionSettingsResponseSettings"
    created_at: datetime.datetime
    updated_at: datetime.datetime
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        sso_connection = self.sso_connection

        settings = self.settings.to_dict()

        created_at = self.created_at.isoformat()

        updated_at = self.updated_at.isoformat()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "sso_connection": sso_connection,
                "settings": settings,
                "created_at": created_at,
                "updated_at": updated_at,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        from ..models.sso_connection_settings_response_settings import SSOConnectionSettingsResponseSettings

        d = src_dict.copy()
        sso_connection = d.pop("sso_connection")

        settings = SSOConnectionSettingsResponseSettings.from_dict(d.pop("settings"))

        created_at = isoparse(d.pop("created_at"))

        updated_at = isoparse(d.pop("updated_at"))

        sso_connection_settings_response = cls(
            sso_connection=sso_connection,
            settings=settings,
            created_at=created_at,
            updated_at=updated_at,
        )

        sso_connection_settings_response.additional_properties = d
        return sso_connection_settings_response

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

import datetime
from typing import TYPE_CHECKING, Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from dateutil.parser import isoparse

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.upsert_sso_connection_settings_request_settings import UpsertSSOConnectionSettingsRequestSettings


T = TypeVar("T", bound="UpsertSSOConnectionSettingsRequest")


@_attrs_define
class UpsertSSOConnectionSettingsRequest:
    """
    Attributes:
        settings (Union[Unset, UpsertSSOConnectionSettingsRequestSettings]):
        expected_updated_at (Union[Unset, datetime.datetime]):
        create_only (Union[Unset, bool]):  Default: False.
    """

    settings: Union[Unset, "UpsertSSOConnectionSettingsRequestSettings"] = UNSET
    expected_updated_at: Union[Unset, datetime.datetime] = UNSET
    create_only: Union[Unset, bool] = False
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        settings: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.settings, Unset):
            settings = self.settings.to_dict()

        expected_updated_at: Union[Unset, str] = UNSET
        if not isinstance(self.expected_updated_at, Unset):
            expected_updated_at = self.expected_updated_at.isoformat()

        create_only = self.create_only

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if settings is not UNSET:
            field_dict["settings"] = settings
        if expected_updated_at is not UNSET:
            field_dict["expected_updated_at"] = expected_updated_at
        if create_only is not UNSET:
            field_dict["create_only"] = create_only

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        from ..models.upsert_sso_connection_settings_request_settings import UpsertSSOConnectionSettingsRequestSettings

        d = src_dict.copy()
        _settings = d.pop("settings", UNSET)
        settings: Union[Unset, UpsertSSOConnectionSettingsRequestSettings]
        if isinstance(_settings, Unset):
            settings = UNSET
        else:
            settings = UpsertSSOConnectionSettingsRequestSettings.from_dict(_settings)

        _expected_updated_at = d.pop("expected_updated_at", UNSET)
        expected_updated_at: Union[Unset, datetime.datetime]
        if isinstance(_expected_updated_at, Unset):
            expected_updated_at = UNSET
        else:
            expected_updated_at = isoparse(_expected_updated_at)

        create_only = d.pop("create_only", UNSET)

        upsert_sso_connection_settings_request = cls(
            settings=settings,
            expected_updated_at=expected_updated_at,
            create_only=create_only,
        )

        upsert_sso_connection_settings_request.additional_properties = d
        return upsert_sso_connection_settings_request

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

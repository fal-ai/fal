from typing import Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="RotateSecretsRequest")


@_attrs_define
class RotateSecretsRequest:
    """
    Attributes:
        reason (Union[Unset, str]):
        delay_old_secrets_expiration_hours (Union[Unset, float]):
    """

    reason: Union[Unset, str] = UNSET
    delay_old_secrets_expiration_hours: Union[Unset, float] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        reason = self.reason

        delay_old_secrets_expiration_hours = self.delay_old_secrets_expiration_hours

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if reason is not UNSET:
            field_dict["reason"] = reason
        if delay_old_secrets_expiration_hours is not UNSET:
            field_dict["delayOldSecretsExpirationHours"] = delay_old_secrets_expiration_hours

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        reason = d.pop("reason", UNSET)

        delay_old_secrets_expiration_hours = d.pop("delayOldSecretsExpirationHours", UNSET)

        rotate_secrets_request = cls(
            reason=reason,
            delay_old_secrets_expiration_hours=delay_old_secrets_expiration_hours,
        )

        rotate_secrets_request.additional_properties = d
        return rotate_secrets_request

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

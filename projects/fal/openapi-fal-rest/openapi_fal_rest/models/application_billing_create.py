from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.app_billing_status import AppBillingStatus

T = TypeVar("T", bound="ApplicationBillingCreate")


@_attrs_define
class ApplicationBillingCreate:
    """
    Attributes:
        user_nickname (str):
        application_alias (str):
        billing_status (AppBillingStatus):
    """

    user_nickname: str
    application_alias: str
    billing_status: AppBillingStatus
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        user_nickname = self.user_nickname

        application_alias = self.application_alias

        billing_status = self.billing_status.value

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "user_nickname": user_nickname,
                "application_alias": application_alias,
                "billing_status": billing_status,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        user_nickname = d.pop("user_nickname")

        application_alias = d.pop("application_alias")

        billing_status = AppBillingStatus(d.pop("billing_status"))

        application_billing_create = cls(
            user_nickname=user_nickname,
            application_alias=application_alias,
            billing_status=billing_status,
        )

        application_billing_create.additional_properties = d
        return application_billing_create

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

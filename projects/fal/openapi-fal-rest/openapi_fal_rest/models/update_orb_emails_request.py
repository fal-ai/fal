from typing import Any, TypeVar, Union, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="UpdateOrbEmailsRequest")


@_attrs_define
class UpdateOrbEmailsRequest:
    """
    Attributes:
        user_id (str):
        primary_billing_email (Union[Unset, str]):
        additional_invoice_emails (Union[Unset, list[str]]):
    """

    user_id: str
    primary_billing_email: Union[Unset, str] = UNSET
    additional_invoice_emails: Union[Unset, list[str]] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        user_id = self.user_id

        primary_billing_email = self.primary_billing_email

        additional_invoice_emails: Union[Unset, list[str]] = UNSET
        if not isinstance(self.additional_invoice_emails, Unset):
            additional_invoice_emails = self.additional_invoice_emails

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "user_id": user_id,
            }
        )
        if primary_billing_email is not UNSET:
            field_dict["primary_billing_email"] = primary_billing_email
        if additional_invoice_emails is not UNSET:
            field_dict["additional_invoice_emails"] = additional_invoice_emails

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        user_id = d.pop("user_id")

        primary_billing_email = d.pop("primary_billing_email", UNSET)

        additional_invoice_emails = cast(list[str], d.pop("additional_invoice_emails", UNSET))

        update_orb_emails_request = cls(
            user_id=user_id,
            primary_billing_email=primary_billing_email,
            additional_invoice_emails=additional_invoice_emails,
        )

        update_orb_emails_request.additional_properties = d
        return update_orb_emails_request

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

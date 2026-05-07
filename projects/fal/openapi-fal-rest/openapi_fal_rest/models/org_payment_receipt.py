import datetime
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from dateutil.parser import isoparse

T = TypeVar("T", bound="OrgPaymentReceipt")


@_attrs_define
class OrgPaymentReceipt:
    """Payment receipt with user_id for organization context.

    Attributes:
        amount (int):
        url (str):
        created (datetime.datetime):
        user_id (str):
    """

    amount: int
    url: str
    created: datetime.datetime
    user_id: str
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        amount = self.amount

        url = self.url

        created = self.created.isoformat()

        user_id = self.user_id

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "amount": amount,
                "url": url,
                "created": created,
                "user_id": user_id,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        amount = d.pop("amount")

        url = d.pop("url")

        created = isoparse(d.pop("created"))

        user_id = d.pop("user_id")

        org_payment_receipt = cls(
            amount=amount,
            url=url,
            created=created,
            user_id=user_id,
        )

        org_payment_receipt.additional_properties = d
        return org_payment_receipt

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

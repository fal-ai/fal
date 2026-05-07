from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

T = TypeVar("T", bound="CreditPurchaseInvoice")


@_attrs_define
class CreditPurchaseInvoice:
    """
    Attributes:
        id (str):
        invoice_date (str):
        total (str):
        hosted_invoice_url (str):
    """

    id: str
    invoice_date: str
    total: str
    hosted_invoice_url: str
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        id = self.id

        invoice_date = self.invoice_date

        total = self.total

        hosted_invoice_url = self.hosted_invoice_url

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "id": id,
                "invoice_date": invoice_date,
                "total": total,
                "hosted_invoice_url": hosted_invoice_url,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        id = d.pop("id")

        invoice_date = d.pop("invoice_date")

        total = d.pop("total")

        hosted_invoice_url = d.pop("hosted_invoice_url")

        credit_purchase_invoice = cls(
            id=id,
            invoice_date=invoice_date,
            total=total,
            hosted_invoice_url=hosted_invoice_url,
        )

        credit_purchase_invoice.additional_properties = d
        return credit_purchase_invoice

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

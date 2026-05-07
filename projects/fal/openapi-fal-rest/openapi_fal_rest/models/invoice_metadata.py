from typing import TYPE_CHECKING, Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.invoice_metadata_billing_address import InvoiceMetadataBillingAddress
    from ..models.invoice_metadata_customer_tax_id import InvoiceMetadataCustomerTaxId


T = TypeVar("T", bound="InvoiceMetadata")


@_attrs_define
class InvoiceMetadata:
    """Metadata about the invoice.

    Attributes:
        invoice_id (str):
        invoice_number (str):
        customer_id (str):
        external_customer_id (str):
        invoice_date (str):
        due_date (str):
        amount_due (str):
        currency (str):
        billing_address (Union[Unset, InvoiceMetadataBillingAddress]):
        customer_tax_id (Union[Unset, InvoiceMetadataCustomerTaxId]):
    """

    invoice_id: str
    invoice_number: str
    customer_id: str
    external_customer_id: str
    invoice_date: str
    due_date: str
    amount_due: str
    currency: str
    billing_address: Union[Unset, "InvoiceMetadataBillingAddress"] = UNSET
    customer_tax_id: Union[Unset, "InvoiceMetadataCustomerTaxId"] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        invoice_id = self.invoice_id

        invoice_number = self.invoice_number

        customer_id = self.customer_id

        external_customer_id = self.external_customer_id

        invoice_date = self.invoice_date

        due_date = self.due_date

        amount_due = self.amount_due

        currency = self.currency

        billing_address: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.billing_address, Unset):
            billing_address = self.billing_address.to_dict()

        customer_tax_id: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.customer_tax_id, Unset):
            customer_tax_id = self.customer_tax_id.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "invoice_id": invoice_id,
                "invoice_number": invoice_number,
                "customer_id": customer_id,
                "external_customer_id": external_customer_id,
                "invoice_date": invoice_date,
                "due_date": due_date,
                "amount_due": amount_due,
                "currency": currency,
            }
        )
        if billing_address is not UNSET:
            field_dict["billing_address"] = billing_address
        if customer_tax_id is not UNSET:
            field_dict["customer_tax_id"] = customer_tax_id

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        from ..models.invoice_metadata_billing_address import InvoiceMetadataBillingAddress
        from ..models.invoice_metadata_customer_tax_id import InvoiceMetadataCustomerTaxId

        d = src_dict.copy()
        invoice_id = d.pop("invoice_id")

        invoice_number = d.pop("invoice_number")

        customer_id = d.pop("customer_id")

        external_customer_id = d.pop("external_customer_id")

        invoice_date = d.pop("invoice_date")

        due_date = d.pop("due_date")

        amount_due = d.pop("amount_due")

        currency = d.pop("currency")

        _billing_address = d.pop("billing_address", UNSET)
        billing_address: Union[Unset, InvoiceMetadataBillingAddress]
        if isinstance(_billing_address, Unset):
            billing_address = UNSET
        else:
            billing_address = InvoiceMetadataBillingAddress.from_dict(_billing_address)

        _customer_tax_id = d.pop("customer_tax_id", UNSET)
        customer_tax_id: Union[Unset, InvoiceMetadataCustomerTaxId]
        if isinstance(_customer_tax_id, Unset):
            customer_tax_id = UNSET
        else:
            customer_tax_id = InvoiceMetadataCustomerTaxId.from_dict(_customer_tax_id)

        invoice_metadata = cls(
            invoice_id=invoice_id,
            invoice_number=invoice_number,
            customer_id=customer_id,
            external_customer_id=external_customer_id,
            invoice_date=invoice_date,
            due_date=due_date,
            amount_due=amount_due,
            currency=currency,
            billing_address=billing_address,
            customer_tax_id=customer_tax_id,
        )

        invoice_metadata.additional_properties = d
        return invoice_metadata

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

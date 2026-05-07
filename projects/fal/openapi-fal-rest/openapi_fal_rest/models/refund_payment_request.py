from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

T = TypeVar("T", bound="RefundPaymentRequest")


@_attrs_define
class RefundPaymentRequest:
    """
    Attributes:
        invoice_id (str):
        refund_reason (str):
    """

    invoice_id: str
    refund_reason: str
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        invoice_id = self.invoice_id

        refund_reason = self.refund_reason

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "invoiceId": invoice_id,
                "refundReason": refund_reason,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        invoice_id = d.pop("invoiceId")

        refund_reason = d.pop("refundReason")

        refund_payment_request = cls(
            invoice_id=invoice_id,
            refund_reason=refund_reason,
        )

        refund_payment_request.additional_properties = d
        return refund_payment_request

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

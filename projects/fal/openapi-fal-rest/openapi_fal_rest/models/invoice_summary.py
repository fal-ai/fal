from typing import TYPE_CHECKING, Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.invoice_summary_subtotal_by_section import InvoiceSummarySubtotalBySection


T = TypeVar("T", bound="InvoiceSummary")


@_attrs_define
class InvoiceSummary:
    """Summary information for the invoice.

    Attributes:
        total_amount (str):
        subtotal_by_section (InvoiceSummarySubtotalBySection):
        total_platform_discount (float):
        total_endpoint_discount (Union[Unset, float]):  Default: 0.0.
    """

    total_amount: str
    subtotal_by_section: "InvoiceSummarySubtotalBySection"
    total_platform_discount: float
    total_endpoint_discount: Union[Unset, float] = 0.0
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        total_amount = self.total_amount

        subtotal_by_section = self.subtotal_by_section.to_dict()

        total_platform_discount = self.total_platform_discount

        total_endpoint_discount = self.total_endpoint_discount

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "total_amount": total_amount,
                "subtotal_by_section": subtotal_by_section,
                "total_platform_discount": total_platform_discount,
            }
        )
        if total_endpoint_discount is not UNSET:
            field_dict["total_endpoint_discount"] = total_endpoint_discount

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        from ..models.invoice_summary_subtotal_by_section import InvoiceSummarySubtotalBySection

        d = src_dict.copy()
        total_amount = d.pop("total_amount")

        subtotal_by_section = InvoiceSummarySubtotalBySection.from_dict(d.pop("subtotal_by_section"))

        total_platform_discount = d.pop("total_platform_discount")

        total_endpoint_discount = d.pop("total_endpoint_discount", UNSET)

        invoice_summary = cls(
            total_amount=total_amount,
            subtotal_by_section=subtotal_by_section,
            total_platform_discount=total_platform_discount,
            total_endpoint_discount=total_endpoint_discount,
        )

        invoice_summary.additional_properties = d
        return invoice_summary

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

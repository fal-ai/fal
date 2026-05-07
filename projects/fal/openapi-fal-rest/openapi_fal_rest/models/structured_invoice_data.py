from typing import TYPE_CHECKING, Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.invoice_metadata import InvoiceMetadata
    from ..models.invoice_section import InvoiceSection
    from ..models.invoice_summary import InvoiceSummary
    from ..models.team_usage import TeamUsage


T = TypeVar("T", bound="StructuredInvoiceData")


@_attrs_define
class StructuredInvoiceData:
    """Complete structured invoice data with enrichment.

    Attributes:
        invoice_metadata (InvoiceMetadata): Metadata about the invoice.
        sections (list['InvoiceSection']):
        summary (InvoiceSummary): Summary information for the invoice.
        team_breakdown (Union[Unset, list['TeamUsage']]):
    """

    invoice_metadata: "InvoiceMetadata"
    sections: list["InvoiceSection"]
    summary: "InvoiceSummary"
    team_breakdown: Union[Unset, list["TeamUsage"]] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        invoice_metadata = self.invoice_metadata.to_dict()

        sections = []
        for sections_item_data in self.sections:
            sections_item = sections_item_data.to_dict()
            sections.append(sections_item)

        summary = self.summary.to_dict()

        team_breakdown: Union[Unset, list[dict[str, Any]]] = UNSET
        if not isinstance(self.team_breakdown, Unset):
            team_breakdown = []
            for team_breakdown_item_data in self.team_breakdown:
                team_breakdown_item = team_breakdown_item_data.to_dict()
                team_breakdown.append(team_breakdown_item)

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "invoice_metadata": invoice_metadata,
                "sections": sections,
                "summary": summary,
            }
        )
        if team_breakdown is not UNSET:
            field_dict["team_breakdown"] = team_breakdown

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        from ..models.invoice_metadata import InvoiceMetadata
        from ..models.invoice_section import InvoiceSection
        from ..models.invoice_summary import InvoiceSummary
        from ..models.team_usage import TeamUsage

        d = src_dict.copy()
        invoice_metadata = InvoiceMetadata.from_dict(d.pop("invoice_metadata"))

        sections = []
        _sections = d.pop("sections")
        for sections_item_data in _sections:
            sections_item = InvoiceSection.from_dict(sections_item_data)

            sections.append(sections_item)

        summary = InvoiceSummary.from_dict(d.pop("summary"))

        team_breakdown = []
        _team_breakdown = d.pop("team_breakdown", UNSET)
        for team_breakdown_item_data in _team_breakdown or []:
            team_breakdown_item = TeamUsage.from_dict(team_breakdown_item_data)

            team_breakdown.append(team_breakdown_item)

        structured_invoice_data = cls(
            invoice_metadata=invoice_metadata,
            sections=sections,
            summary=summary,
            team_breakdown=team_breakdown,
        )

        structured_invoice_data.additional_properties = d
        return structured_invoice_data

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

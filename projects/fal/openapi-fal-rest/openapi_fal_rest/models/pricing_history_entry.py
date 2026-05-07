import datetime
from typing import TYPE_CHECKING, Any, TypeVar, Union
from uuid import UUID

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from dateutil.parser import isoparse

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.pricing_history_entry_attributes import PricingHistoryEntryAttributes


T = TypeVar("T", bound="PricingHistoryEntry")


@_attrs_define
class PricingHistoryEntry:
    """
    Attributes:
        event_id (UUID):
        pricing_type (str):
        pricing_key (str):
        unit_price (float):
        billing_unit (str):
        effective_at (datetime.datetime):
        operation (str):
        attributes (Union[Unset, PricingHistoryEntryAttributes]):
    """

    event_id: UUID
    pricing_type: str
    pricing_key: str
    unit_price: float
    billing_unit: str
    effective_at: datetime.datetime
    operation: str
    attributes: Union[Unset, "PricingHistoryEntryAttributes"] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        event_id = str(self.event_id)

        pricing_type = self.pricing_type

        pricing_key = self.pricing_key

        unit_price = self.unit_price

        billing_unit = self.billing_unit

        effective_at = self.effective_at.isoformat()

        operation = self.operation

        attributes: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.attributes, Unset):
            attributes = self.attributes.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "event_id": event_id,
                "pricing_type": pricing_type,
                "pricing_key": pricing_key,
                "unit_price": unit_price,
                "billing_unit": billing_unit,
                "effective_at": effective_at,
                "operation": operation,
            }
        )
        if attributes is not UNSET:
            field_dict["attributes"] = attributes

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        from ..models.pricing_history_entry_attributes import PricingHistoryEntryAttributes

        d = src_dict.copy()
        event_id = UUID(d.pop("event_id"))

        pricing_type = d.pop("pricing_type")

        pricing_key = d.pop("pricing_key")

        unit_price = d.pop("unit_price")

        billing_unit = d.pop("billing_unit")

        effective_at = isoparse(d.pop("effective_at"))

        operation = d.pop("operation")

        _attributes = d.pop("attributes", UNSET)
        attributes: Union[Unset, PricingHistoryEntryAttributes]
        if isinstance(_attributes, Unset):
            attributes = UNSET
        else:
            attributes = PricingHistoryEntryAttributes.from_dict(_attributes)

        pricing_history_entry = cls(
            event_id=event_id,
            pricing_type=pricing_type,
            pricing_key=pricing_key,
            unit_price=unit_price,
            billing_unit=billing_unit,
            effective_at=effective_at,
            operation=operation,
            attributes=attributes,
        )

        pricing_history_entry.additional_properties = d
        return pricing_history_entry

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

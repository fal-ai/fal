from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

T = TypeVar("T", bound="OrgBilling")


@_attrs_define
class OrgBilling:
    """Org-wide billing summary for org admins/billing users.

    Attributes:
        has_invoicing: Whether any team in the org uses invoicing.

        Attributes:
            has_invoicing (bool):
    """

    has_invoicing: bool
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        has_invoicing = self.has_invoicing

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "has_invoicing": has_invoicing,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        has_invoicing = d.pop("has_invoicing")

        org_billing = cls(
            has_invoicing=has_invoicing,
        )

        org_billing.additional_properties = d
        return org_billing

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

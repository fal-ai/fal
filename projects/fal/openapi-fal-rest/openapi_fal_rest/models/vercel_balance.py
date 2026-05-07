from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

T = TypeVar("T", bound="VercelBalance")


@_attrs_define
class VercelBalance:
    """
    Attributes:
        currency_value_in_cents (int):
        credit (str):
        name_label (str):
    """

    currency_value_in_cents: int
    credit: str
    name_label: str
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        currency_value_in_cents = self.currency_value_in_cents

        credit = self.credit

        name_label = self.name_label

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "currencyValueInCents": currency_value_in_cents,
                "credit": credit,
                "nameLabel": name_label,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        currency_value_in_cents = d.pop("currencyValueInCents")

        credit = d.pop("credit")

        name_label = d.pop("nameLabel")

        vercel_balance = cls(
            currency_value_in_cents=currency_value_in_cents,
            credit=credit,
            name_label=name_label,
        )

        vercel_balance.additional_properties = d
        return vercel_balance

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

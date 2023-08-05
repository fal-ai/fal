from typing import Any, Dict, List, Type, TypeVar

import attr

T = TypeVar("T", bound="PaymentMethod")


@attr.s(auto_attribs=True)
class PaymentMethod:
    """
    Attributes:
        type (str):
        brand (str):
        last4 (str):
        exp_month (int):
        exp_year (int):
    """

    type: str
    brand: str
    last4: str
    exp_month: int
    exp_year: int
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        type = self.type
        brand = self.brand
        last4 = self.last4
        exp_month = self.exp_month
        exp_year = self.exp_year

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "type": type,
                "brand": brand,
                "last4": last4,
                "exp_month": exp_month,
                "exp_year": exp_year,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        type = d.pop("type")

        brand = d.pop("brand")

        last4 = d.pop("last4")

        exp_month = d.pop("exp_month")

        exp_year = d.pop("exp_year")

        payment_method = cls(
            type=type,
            brand=brand,
            last4=last4,
            exp_month=exp_month,
            exp_year=exp_year,
        )

        payment_method.additional_properties = d
        return payment_method

    @property
    def additional_keys(self) -> List[str]:
        return list(self.additional_properties.keys())

    def __getitem__(self, key: str) -> Any:
        return self.additional_properties[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.additional_properties[key] = value

    def __delitem__(self, key: str) -> None:
        del self.additional_properties[key]

    def __contains__(self, key: str) -> bool:
        return key in self.additional_properties

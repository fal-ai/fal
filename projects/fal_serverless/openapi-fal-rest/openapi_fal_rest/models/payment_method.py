from typing import Any, Dict, List, Type, TypeVar, Union

import attr

from ..types import UNSET, Unset

T = TypeVar("T", bound="PaymentMethod")


@attr.s(auto_attribs=True)
class PaymentMethod:
    """
    Attributes:
        id (str):
        type (str):
        last4 (str):
        brand (str):
        exp_month (Union[Unset, int]):
        exp_year (Union[Unset, int]):
    """

    id: str
    type: str
    last4: str
    brand: str
    exp_month: Union[Unset, int] = UNSET
    exp_year: Union[Unset, int] = UNSET
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        id = self.id
        type = self.type
        last4 = self.last4
        brand = self.brand
        exp_month = self.exp_month
        exp_year = self.exp_year

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "id": id,
                "type": type,
                "last4": last4,
                "brand": brand,
            }
        )
        if exp_month is not UNSET:
            field_dict["exp_month"] = exp_month
        if exp_year is not UNSET:
            field_dict["exp_year"] = exp_year

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        id = d.pop("id")

        type = d.pop("type")

        last4 = d.pop("last4")

        brand = d.pop("brand")

        exp_month = d.pop("exp_month", UNSET)

        exp_year = d.pop("exp_year", UNSET)

        payment_method = cls(
            id=id,
            type=type,
            last4=last4,
            brand=brand,
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

from typing import Any, Dict, List, Type, TypeVar, Union

import attr

from ..types import UNSET, Unset

T = TypeVar("T", bound="CustomerDetails")


@attr.s(auto_attribs=True)
class CustomerDetails:
    """
    Attributes:
        user_id (str):
        soft_monthly_budget (Union[Unset, int]):  Default: 2500.
        hard_monthly_budget (Union[Unset, int]):  Default: 3000.
        current_balance (Union[Unset, int]):
    """

    user_id: str
    soft_monthly_budget: Union[Unset, int] = 2500
    hard_monthly_budget: Union[Unset, int] = 3000
    current_balance: Union[Unset, int] = 0
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        user_id = self.user_id
        soft_monthly_budget = self.soft_monthly_budget
        hard_monthly_budget = self.hard_monthly_budget
        current_balance = self.current_balance

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "user_id": user_id,
            }
        )
        if soft_monthly_budget is not UNSET:
            field_dict["soft_monthly_budget"] = soft_monthly_budget
        if hard_monthly_budget is not UNSET:
            field_dict["hard_monthly_budget"] = hard_monthly_budget
        if current_balance is not UNSET:
            field_dict["current_balance"] = current_balance

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        user_id = d.pop("user_id")

        soft_monthly_budget = d.pop("soft_monthly_budget", UNSET)

        hard_monthly_budget = d.pop("hard_monthly_budget", UNSET)

        current_balance = d.pop("current_balance", UNSET)

        customer_details = cls(
            user_id=user_id,
            soft_monthly_budget=soft_monthly_budget,
            hard_monthly_budget=hard_monthly_budget,
            current_balance=current_balance,
        )

        customer_details.additional_properties = d
        return customer_details

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

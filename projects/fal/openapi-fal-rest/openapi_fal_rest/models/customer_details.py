from typing import Any, Dict, List, Type, TypeVar, Union

import attr

from ..models.lock_reason import LockReason
from ..types import UNSET, Unset

T = TypeVar("T", bound="CustomerDetails")


@attr.s(auto_attribs=True)
class CustomerDetails:
    """
    Attributes:
        user_id (str):
        soft_monthly_budget (Union[Unset, int]):
        hard_monthly_budget (Union[Unset, int]):
        current_balance (Union[Unset, int]):
        is_paying (Union[Unset, bool]):
        is_locked (Union[Unset, bool]):
        lock_reason (Union[Unset, None, LockReason]): An enumeration.
    """

    user_id: str
    soft_monthly_budget: Union[Unset, int] = UNSET
    hard_monthly_budget: Union[Unset, int] = UNSET
    current_balance: Union[Unset, int] = 0
    is_paying: Union[Unset, bool] = False
    is_locked: Union[Unset, bool] = False
    lock_reason: Union[Unset, None, LockReason] = UNSET
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        user_id = self.user_id
        soft_monthly_budget = self.soft_monthly_budget
        hard_monthly_budget = self.hard_monthly_budget
        current_balance = self.current_balance
        is_paying = self.is_paying
        is_locked = self.is_locked
        lock_reason: Union[Unset, None, str] = UNSET
        if not isinstance(self.lock_reason, Unset):
            lock_reason = self.lock_reason.value if self.lock_reason else None

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
        if is_paying is not UNSET:
            field_dict["is_paying"] = is_paying
        if is_locked is not UNSET:
            field_dict["is_locked"] = is_locked
        if lock_reason is not UNSET:
            field_dict["lock_reason"] = lock_reason

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        user_id = d.pop("user_id")

        soft_monthly_budget = d.pop("soft_monthly_budget", UNSET)

        hard_monthly_budget = d.pop("hard_monthly_budget", UNSET)

        current_balance = d.pop("current_balance", UNSET)

        is_paying = d.pop("is_paying", UNSET)

        is_locked = d.pop("is_locked", UNSET)

        _lock_reason = d.pop("lock_reason", UNSET)
        lock_reason: Union[Unset, None, LockReason]
        if _lock_reason is None:
            lock_reason = None
        elif isinstance(_lock_reason, Unset):
            lock_reason = UNSET
        else:
            lock_reason = LockReason(_lock_reason)

        customer_details = cls(
            user_id=user_id,
            soft_monthly_budget=soft_monthly_budget,
            hard_monthly_budget=hard_monthly_budget,
            current_balance=current_balance,
            is_paying=is_paying,
            is_locked=is_locked,
            lock_reason=lock_reason,
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

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
        lock_reason (Union[Unset, None, LockReason]): An enumeration.
        is_paying (Union[Unset, bool]):
        is_locked (Union[Unset, bool]):
        is_eligible_for_extra_credits (Union[Unset, bool]):
    """

    user_id: str
    soft_monthly_budget: Union[Unset, int] = UNSET
    hard_monthly_budget: Union[Unset, int] = UNSET
    lock_reason: Union[Unset, None, LockReason] = UNSET
    is_paying: Union[Unset, bool] = False
    is_locked: Union[Unset, bool] = False
    is_eligible_for_extra_credits: Union[Unset, bool] = False
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        user_id = self.user_id
        soft_monthly_budget = self.soft_monthly_budget
        hard_monthly_budget = self.hard_monthly_budget
        lock_reason: Union[Unset, None, str] = UNSET
        if not isinstance(self.lock_reason, Unset):
            lock_reason = self.lock_reason.value if self.lock_reason else None

        is_paying = self.is_paying
        is_locked = self.is_locked
        is_eligible_for_extra_credits = self.is_eligible_for_extra_credits

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
        if lock_reason is not UNSET:
            field_dict["lock_reason"] = lock_reason
        if is_paying is not UNSET:
            field_dict["is_paying"] = is_paying
        if is_locked is not UNSET:
            field_dict["is_locked"] = is_locked
        if is_eligible_for_extra_credits is not UNSET:
            field_dict["is_eligible_for_extra_credits"] = is_eligible_for_extra_credits

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        user_id = d.pop("user_id")

        soft_monthly_budget = d.pop("soft_monthly_budget", UNSET)

        hard_monthly_budget = d.pop("hard_monthly_budget", UNSET)

        _lock_reason = d.pop("lock_reason", UNSET)
        lock_reason: Union[Unset, None, LockReason]
        if _lock_reason is None:
            lock_reason = None
        elif isinstance(_lock_reason, Unset):
            lock_reason = UNSET
        else:
            lock_reason = LockReason(_lock_reason)

        is_paying = d.pop("is_paying", UNSET)

        is_locked = d.pop("is_locked", UNSET)

        is_eligible_for_extra_credits = d.pop("is_eligible_for_extra_credits", UNSET)

        customer_details = cls(
            user_id=user_id,
            soft_monthly_budget=soft_monthly_budget,
            hard_monthly_budget=hard_monthly_budget,
            lock_reason=lock_reason,
            is_paying=is_paying,
            is_locked=is_locked,
            is_eligible_for_extra_credits=is_eligible_for_extra_credits,
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

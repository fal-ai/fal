from typing import Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.lock_reason import LockReason
from ..models.payment_verification_status import PaymentVerificationStatus
from ..types import UNSET, Unset

T = TypeVar("T", bound="CustomerDetails")


@_attrs_define
class CustomerDetails:
    """
    Attributes:
        user_id (str):
        hard_monthly_budget (int):
        lock_reason (LockReason):
        is_invoicing (Union[Unset, bool]):  Default: False.
        is_locked (Union[Unset, bool]):  Default: False.
        is_eligible_for_extra_credits (Union[Unset, bool]):  Default: False.
        payment_verification_status (Union[Unset, PaymentVerificationStatus]):
    """

    user_id: str
    hard_monthly_budget: int
    lock_reason: LockReason
    is_invoicing: Union[Unset, bool] = False
    is_locked: Union[Unset, bool] = False
    is_eligible_for_extra_credits: Union[Unset, bool] = False
    payment_verification_status: Union[Unset, PaymentVerificationStatus] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        user_id = self.user_id

        hard_monthly_budget = self.hard_monthly_budget

        lock_reason = self.lock_reason.value

        is_invoicing = self.is_invoicing

        is_locked = self.is_locked

        is_eligible_for_extra_credits = self.is_eligible_for_extra_credits

        payment_verification_status: Union[Unset, str] = UNSET
        if not isinstance(self.payment_verification_status, Unset):
            payment_verification_status = self.payment_verification_status.value

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "user_id": user_id,
                "hard_monthly_budget": hard_monthly_budget,
                "lock_reason": lock_reason,
            }
        )
        if is_invoicing is not UNSET:
            field_dict["is_invoicing"] = is_invoicing
        if is_locked is not UNSET:
            field_dict["is_locked"] = is_locked
        if is_eligible_for_extra_credits is not UNSET:
            field_dict["is_eligible_for_extra_credits"] = is_eligible_for_extra_credits
        if payment_verification_status is not UNSET:
            field_dict["payment_verification_status"] = payment_verification_status

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        user_id = d.pop("user_id")

        hard_monthly_budget = d.pop("hard_monthly_budget")

        lock_reason = LockReason(d.pop("lock_reason"))

        is_invoicing = d.pop("is_invoicing", UNSET)

        is_locked = d.pop("is_locked", UNSET)

        is_eligible_for_extra_credits = d.pop("is_eligible_for_extra_credits", UNSET)

        _payment_verification_status = d.pop("payment_verification_status", UNSET)
        payment_verification_status: Union[Unset, PaymentVerificationStatus]
        if isinstance(_payment_verification_status, Unset):
            payment_verification_status = UNSET
        else:
            payment_verification_status = PaymentVerificationStatus(_payment_verification_status)

        customer_details = cls(
            user_id=user_id,
            hard_monthly_budget=hard_monthly_budget,
            lock_reason=lock_reason,
            is_invoicing=is_invoicing,
            is_locked=is_locked,
            is_eligible_for_extra_credits=is_eligible_for_extra_credits,
            payment_verification_status=payment_verification_status,
        )

        customer_details.additional_properties = d
        return customer_details

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

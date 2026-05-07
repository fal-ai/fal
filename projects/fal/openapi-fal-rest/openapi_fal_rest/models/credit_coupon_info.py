import datetime
from typing import Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from dateutil.parser import isoparse

from ..types import UNSET, Unset

T = TypeVar("T", bound="CreditCouponInfo")


@_attrs_define
class CreditCouponInfo:
    """
    Attributes:
        coupon_id (str):
        claim_limit (int):
        amount_dollars (int):
        start_date (Union[Unset, datetime.datetime]):
        end_date (Union[Unset, datetime.datetime]):
        claim_count (Union[Unset, int]):
        expires_in_days (Union[Unset, int]):  Default: 90.
        skip_check (Union[Unset, bool]):  Default: False.
    """

    coupon_id: str
    claim_limit: int
    amount_dollars: int
    start_date: Union[Unset, datetime.datetime] = UNSET
    end_date: Union[Unset, datetime.datetime] = UNSET
    claim_count: Union[Unset, int] = UNSET
    expires_in_days: Union[Unset, int] = 90
    skip_check: Union[Unset, bool] = False
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        coupon_id = self.coupon_id

        claim_limit = self.claim_limit

        amount_dollars = self.amount_dollars

        start_date: Union[Unset, str] = UNSET
        if not isinstance(self.start_date, Unset):
            start_date = self.start_date.isoformat()

        end_date: Union[Unset, str] = UNSET
        if not isinstance(self.end_date, Unset):
            end_date = self.end_date.isoformat()

        claim_count = self.claim_count

        expires_in_days = self.expires_in_days

        skip_check = self.skip_check

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "coupon_id": coupon_id,
                "claim_limit": claim_limit,
                "amount_dollars": amount_dollars,
            }
        )
        if start_date is not UNSET:
            field_dict["start_date"] = start_date
        if end_date is not UNSET:
            field_dict["end_date"] = end_date
        if claim_count is not UNSET:
            field_dict["claim_count"] = claim_count
        if expires_in_days is not UNSET:
            field_dict["expires_in_days"] = expires_in_days
        if skip_check is not UNSET:
            field_dict["skip_check"] = skip_check

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        coupon_id = d.pop("coupon_id")

        claim_limit = d.pop("claim_limit")

        amount_dollars = d.pop("amount_dollars")

        _start_date = d.pop("start_date", UNSET)
        start_date: Union[Unset, datetime.datetime]
        if isinstance(_start_date, Unset):
            start_date = UNSET
        else:
            start_date = isoparse(_start_date)

        _end_date = d.pop("end_date", UNSET)
        end_date: Union[Unset, datetime.datetime]
        if isinstance(_end_date, Unset):
            end_date = UNSET
        else:
            end_date = isoparse(_end_date)

        claim_count = d.pop("claim_count", UNSET)

        expires_in_days = d.pop("expires_in_days", UNSET)

        skip_check = d.pop("skip_check", UNSET)

        credit_coupon_info = cls(
            coupon_id=coupon_id,
            claim_limit=claim_limit,
            amount_dollars=amount_dollars,
            start_date=start_date,
            end_date=end_date,
            claim_count=claim_count,
            expires_in_days=expires_in_days,
            skip_check=skip_check,
        )

        credit_coupon_info.additional_properties = d
        return credit_coupon_info

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

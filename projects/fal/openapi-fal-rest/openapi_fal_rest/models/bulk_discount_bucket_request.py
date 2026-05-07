import datetime
from typing import Any, TypeVar, Union, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from dateutil.parser import isoparse

from ..models.discount_bucket_type import DiscountBucketType
from ..types import UNSET, Unset

T = TypeVar("T", bound="BulkDiscountBucketRequest")


@_attrs_define
class BulkDiscountBucketRequest:
    """
    Attributes:
        user_id (str):
        products (list[str]):
        discount_bucket (str):
        effective_date (datetime.date):
        type_ (Union[Unset, DiscountBucketType]):
        effective_end_date (Union[Unset, datetime.date]):
        replace_existing (Union[Unset, bool]):  Default: False.
    """

    user_id: str
    products: list[str]
    discount_bucket: str
    effective_date: datetime.date
    type_: Union[Unset, DiscountBucketType] = UNSET
    effective_end_date: Union[Unset, datetime.date] = UNSET
    replace_existing: Union[Unset, bool] = False
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        user_id = self.user_id

        products = self.products

        discount_bucket = self.discount_bucket

        effective_date = self.effective_date.isoformat()

        type_: Union[Unset, str] = UNSET
        if not isinstance(self.type_, Unset):
            type_ = self.type_.value

        effective_end_date: Union[Unset, str] = UNSET
        if not isinstance(self.effective_end_date, Unset):
            effective_end_date = self.effective_end_date.isoformat()

        replace_existing = self.replace_existing

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "user_id": user_id,
                "products": products,
                "discount_bucket": discount_bucket,
                "effective_date": effective_date,
            }
        )
        if type_ is not UNSET:
            field_dict["type"] = type_
        if effective_end_date is not UNSET:
            field_dict["effective_end_date"] = effective_end_date
        if replace_existing is not UNSET:
            field_dict["replace_existing"] = replace_existing

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        user_id = d.pop("user_id")

        products = cast(list[str], d.pop("products"))

        discount_bucket = d.pop("discount_bucket")

        effective_date = isoparse(d.pop("effective_date")).date()

        _type_ = d.pop("type", UNSET)
        type_: Union[Unset, DiscountBucketType]
        if isinstance(_type_, Unset):
            type_ = UNSET
        else:
            type_ = DiscountBucketType(_type_)

        _effective_end_date = d.pop("effective_end_date", UNSET)
        effective_end_date: Union[Unset, datetime.date]
        if isinstance(_effective_end_date, Unset):
            effective_end_date = UNSET
        else:
            effective_end_date = isoparse(_effective_end_date).date()

        replace_existing = d.pop("replace_existing", UNSET)

        bulk_discount_bucket_request = cls(
            user_id=user_id,
            products=products,
            discount_bucket=discount_bucket,
            effective_date=effective_date,
            type_=type_,
            effective_end_date=effective_end_date,
            replace_existing=replace_existing,
        )

        bulk_discount_bucket_request.additional_properties = d
        return bulk_discount_bucket_request

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

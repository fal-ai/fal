import datetime
from typing import Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from dateutil.parser import isoparse

from ..models.discount_bucket_type import DiscountBucketType
from ..types import UNSET, Unset

T = TypeVar("T", bound="CustomerDiscountBucket")


@_attrs_define
class CustomerDiscountBucket:
    """
    Attributes:
        user_id (str):
        product (str):
        discount_bucket (str):
        effective_date (datetime.date):
        type_ (Union[Unset, DiscountBucketType]):
        effective_end_date (Union[Unset, datetime.date]):
        username (Union[Unset, str]):
        created_at (Union[Unset, datetime.datetime]):
        updated_at (Union[Unset, datetime.datetime]):
    """

    user_id: str
    product: str
    discount_bucket: str
    effective_date: datetime.date
    type_: Union[Unset, DiscountBucketType] = UNSET
    effective_end_date: Union[Unset, datetime.date] = UNSET
    username: Union[Unset, str] = UNSET
    created_at: Union[Unset, datetime.datetime] = UNSET
    updated_at: Union[Unset, datetime.datetime] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        user_id = self.user_id

        product = self.product

        discount_bucket = self.discount_bucket

        effective_date = self.effective_date.isoformat()

        type_: Union[Unset, str] = UNSET
        if not isinstance(self.type_, Unset):
            type_ = self.type_.value

        effective_end_date: Union[Unset, str] = UNSET
        if not isinstance(self.effective_end_date, Unset):
            effective_end_date = self.effective_end_date.isoformat()

        username = self.username

        created_at: Union[Unset, str] = UNSET
        if not isinstance(self.created_at, Unset):
            created_at = self.created_at.isoformat()

        updated_at: Union[Unset, str] = UNSET
        if not isinstance(self.updated_at, Unset):
            updated_at = self.updated_at.isoformat()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "user_id": user_id,
                "product": product,
                "discount_bucket": discount_bucket,
                "effective_date": effective_date,
            }
        )
        if type_ is not UNSET:
            field_dict["type"] = type_
        if effective_end_date is not UNSET:
            field_dict["effective_end_date"] = effective_end_date
        if username is not UNSET:
            field_dict["username"] = username
        if created_at is not UNSET:
            field_dict["created_at"] = created_at
        if updated_at is not UNSET:
            field_dict["updated_at"] = updated_at

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        user_id = d.pop("user_id")

        product = d.pop("product")

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

        username = d.pop("username", UNSET)

        _created_at = d.pop("created_at", UNSET)
        created_at: Union[Unset, datetime.datetime]
        if isinstance(_created_at, Unset):
            created_at = UNSET
        else:
            created_at = isoparse(_created_at)

        _updated_at = d.pop("updated_at", UNSET)
        updated_at: Union[Unset, datetime.datetime]
        if isinstance(_updated_at, Unset):
            updated_at = UNSET
        else:
            updated_at = isoparse(_updated_at)

        customer_discount_bucket = cls(
            user_id=user_id,
            product=product,
            discount_bucket=discount_bucket,
            effective_date=effective_date,
            type_=type_,
            effective_end_date=effective_end_date,
            username=username,
            created_at=created_at,
            updated_at=updated_at,
        )

        customer_discount_bucket.additional_properties = d
        return customer_discount_bucket

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

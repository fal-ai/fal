import datetime
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from dateutil.parser import isoparse

T = TypeVar("T", bound="UserDiscount")


@_attrs_define
class UserDiscount:
    """
    Attributes:
        discount_id (str):
        user_id (str):
        percent_discount (float):
        starts_at (datetime.datetime):
        ends_at (datetime.datetime):
    """

    discount_id: str
    user_id: str
    percent_discount: float
    starts_at: datetime.datetime
    ends_at: datetime.datetime
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        discount_id = self.discount_id

        user_id = self.user_id

        percent_discount = self.percent_discount

        starts_at = self.starts_at.isoformat()

        ends_at = self.ends_at.isoformat()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "discount_id": discount_id,
                "user_id": user_id,
                "percent_discount": percent_discount,
                "starts_at": starts_at,
                "ends_at": ends_at,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        discount_id = d.pop("discount_id")

        user_id = d.pop("user_id")

        percent_discount = d.pop("percent_discount")

        starts_at = isoparse(d.pop("starts_at"))

        ends_at = isoparse(d.pop("ends_at"))

        user_discount = cls(
            discount_id=discount_id,
            user_id=user_id,
            percent_discount=percent_discount,
            starts_at=starts_at,
            ends_at=ends_at,
        )

        user_discount.additional_properties = d
        return user_discount

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

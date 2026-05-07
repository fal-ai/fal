import datetime
from typing import Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from dateutil.parser import isoparse

from ..types import UNSET, Unset

T = TypeVar("T", bound="EndpointBillingOverride")


@_attrs_define
class EndpointBillingOverride:
    """
    Attributes:
        user_id (str):
        endpoint (str):
        billing_unit (str):
        is_draft (Union[Unset, bool]):  Default: False.
        price (Union[Unset, float]):
        username (Union[Unset, str]):
        use_compute_seconds (Union[Unset, bool]):  Default: False.
        discountable (Union[Unset, bool]):  Default: False.
        percent_discount (Union[Unset, float]):
        start_date (Union[Unset, datetime.datetime]):
        end_date (Union[Unset, datetime.datetime]):
    """

    user_id: str
    endpoint: str
    billing_unit: str
    is_draft: Union[Unset, bool] = False
    price: Union[Unset, float] = UNSET
    username: Union[Unset, str] = UNSET
    use_compute_seconds: Union[Unset, bool] = False
    discountable: Union[Unset, bool] = False
    percent_discount: Union[Unset, float] = UNSET
    start_date: Union[Unset, datetime.datetime] = UNSET
    end_date: Union[Unset, datetime.datetime] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        user_id = self.user_id

        endpoint = self.endpoint

        billing_unit = self.billing_unit

        is_draft = self.is_draft

        price = self.price

        username = self.username

        use_compute_seconds = self.use_compute_seconds

        discountable = self.discountable

        percent_discount = self.percent_discount

        start_date: Union[Unset, str] = UNSET
        if not isinstance(self.start_date, Unset):
            start_date = self.start_date.isoformat()

        end_date: Union[Unset, str] = UNSET
        if not isinstance(self.end_date, Unset):
            end_date = self.end_date.isoformat()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "user_id": user_id,
                "endpoint": endpoint,
                "billing_unit": billing_unit,
            }
        )
        if is_draft is not UNSET:
            field_dict["is_draft"] = is_draft
        if price is not UNSET:
            field_dict["price"] = price
        if username is not UNSET:
            field_dict["username"] = username
        if use_compute_seconds is not UNSET:
            field_dict["use_compute_seconds"] = use_compute_seconds
        if discountable is not UNSET:
            field_dict["discountable"] = discountable
        if percent_discount is not UNSET:
            field_dict["percent_discount"] = percent_discount
        if start_date is not UNSET:
            field_dict["start_date"] = start_date
        if end_date is not UNSET:
            field_dict["end_date"] = end_date

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        user_id = d.pop("user_id")

        endpoint = d.pop("endpoint")

        billing_unit = d.pop("billing_unit")

        is_draft = d.pop("is_draft", UNSET)

        price = d.pop("price", UNSET)

        username = d.pop("username", UNSET)

        use_compute_seconds = d.pop("use_compute_seconds", UNSET)

        discountable = d.pop("discountable", UNSET)

        percent_discount = d.pop("percent_discount", UNSET)

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

        endpoint_billing_override = cls(
            user_id=user_id,
            endpoint=endpoint,
            billing_unit=billing_unit,
            is_draft=is_draft,
            price=price,
            username=username,
            use_compute_seconds=use_compute_seconds,
            discountable=discountable,
            percent_discount=percent_discount,
            start_date=start_date,
            end_date=end_date,
        )

        endpoint_billing_override.additional_properties = d
        return endpoint_billing_override

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

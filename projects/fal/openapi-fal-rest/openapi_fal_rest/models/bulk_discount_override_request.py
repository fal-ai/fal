import datetime
from typing import Any, TypeVar, Union, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from dateutil.parser import isoparse

from ..types import UNSET, Unset

T = TypeVar("T", bound="BulkDiscountOverrideRequest")


@_attrs_define
class BulkDiscountOverrideRequest:
    """
    Attributes:
        user_id (str):
        endpoints (list[str]):
        percent_discount (float):
        is_draft (Union[Unset, bool]):  Default: False.
        replace_existing (Union[Unset, bool]):  Default: False.
        start_date (Union[Unset, datetime.datetime]):
        end_date (Union[Unset, datetime.datetime]):
    """

    user_id: str
    endpoints: list[str]
    percent_discount: float
    is_draft: Union[Unset, bool] = False
    replace_existing: Union[Unset, bool] = False
    start_date: Union[Unset, datetime.datetime] = UNSET
    end_date: Union[Unset, datetime.datetime] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        user_id = self.user_id

        endpoints = self.endpoints

        percent_discount = self.percent_discount

        is_draft = self.is_draft

        replace_existing = self.replace_existing

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
                "endpoints": endpoints,
                "percent_discount": percent_discount,
            }
        )
        if is_draft is not UNSET:
            field_dict["is_draft"] = is_draft
        if replace_existing is not UNSET:
            field_dict["replace_existing"] = replace_existing
        if start_date is not UNSET:
            field_dict["start_date"] = start_date
        if end_date is not UNSET:
            field_dict["end_date"] = end_date

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        user_id = d.pop("user_id")

        endpoints = cast(list[str], d.pop("endpoints"))

        percent_discount = d.pop("percent_discount")

        is_draft = d.pop("is_draft", UNSET)

        replace_existing = d.pop("replace_existing", UNSET)

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

        bulk_discount_override_request = cls(
            user_id=user_id,
            endpoints=endpoints,
            percent_discount=percent_discount,
            is_draft=is_draft,
            replace_existing=replace_existing,
            start_date=start_date,
            end_date=end_date,
        )

        bulk_discount_override_request.additional_properties = d
        return bulk_discount_override_request

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

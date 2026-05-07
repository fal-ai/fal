import datetime
from typing import Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from dateutil.parser import isoparse

from ..types import UNSET, Unset

T = TypeVar("T", bound="ComputeBillingOverrideDefinition")


@_attrs_define
class ComputeBillingOverrideDefinition:
    """
    Attributes:
        user_id (str):
        instance_type (str):
        unit_price (float):
        start_date (datetime.datetime):
        is_draft (Union[Unset, bool]):  Default: False.
        end_date (Union[Unset, datetime.datetime]):
    """

    user_id: str
    instance_type: str
    unit_price: float
    start_date: datetime.datetime
    is_draft: Union[Unset, bool] = False
    end_date: Union[Unset, datetime.datetime] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        user_id = self.user_id

        instance_type = self.instance_type

        unit_price = self.unit_price

        start_date = self.start_date.isoformat()

        is_draft = self.is_draft

        end_date: Union[Unset, str] = UNSET
        if not isinstance(self.end_date, Unset):
            end_date = self.end_date.isoformat()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "user_id": user_id,
                "instance_type": instance_type,
                "unit_price": unit_price,
                "start_date": start_date,
            }
        )
        if is_draft is not UNSET:
            field_dict["is_draft"] = is_draft
        if end_date is not UNSET:
            field_dict["end_date"] = end_date

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        user_id = d.pop("user_id")

        instance_type = d.pop("instance_type")

        unit_price = d.pop("unit_price")

        start_date = isoparse(d.pop("start_date"))

        is_draft = d.pop("is_draft", UNSET)

        _end_date = d.pop("end_date", UNSET)
        end_date: Union[Unset, datetime.datetime]
        if isinstance(_end_date, Unset):
            end_date = UNSET
        else:
            end_date = isoparse(_end_date)

        compute_billing_override_definition = cls(
            user_id=user_id,
            instance_type=instance_type,
            unit_price=unit_price,
            start_date=start_date,
            is_draft=is_draft,
            end_date=end_date,
        )

        compute_billing_override_definition.additional_properties = d
        return compute_billing_override_definition

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

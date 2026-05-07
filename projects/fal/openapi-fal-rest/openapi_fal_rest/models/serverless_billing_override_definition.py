import datetime
from typing import Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from dateutil.parser import isoparse

from ..types import UNSET, Unset

T = TypeVar("T", bound="ServerlessBillingOverrideDefinition")


@_attrs_define
class ServerlessBillingOverrideDefinition:
    """
    Attributes:
        user_id (str):
        machine_type (str):
        unit_price (float):
        starts_at (datetime.datetime):
        is_draft (Union[Unset, bool]):  Default: False.
        ends_at (Union[Unset, datetime.datetime]):
    """

    user_id: str
    machine_type: str
    unit_price: float
    starts_at: datetime.datetime
    is_draft: Union[Unset, bool] = False
    ends_at: Union[Unset, datetime.datetime] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        user_id = self.user_id

        machine_type = self.machine_type

        unit_price = self.unit_price

        starts_at = self.starts_at.isoformat()

        is_draft = self.is_draft

        ends_at: Union[Unset, str] = UNSET
        if not isinstance(self.ends_at, Unset):
            ends_at = self.ends_at.isoformat()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "user_id": user_id,
                "machine_type": machine_type,
                "unit_price": unit_price,
                "starts_at": starts_at,
            }
        )
        if is_draft is not UNSET:
            field_dict["is_draft"] = is_draft
        if ends_at is not UNSET:
            field_dict["ends_at"] = ends_at

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        user_id = d.pop("user_id")

        machine_type = d.pop("machine_type")

        unit_price = d.pop("unit_price")

        starts_at = isoparse(d.pop("starts_at"))

        is_draft = d.pop("is_draft", UNSET)

        _ends_at = d.pop("ends_at", UNSET)
        ends_at: Union[Unset, datetime.datetime]
        if isinstance(_ends_at, Unset):
            ends_at = UNSET
        else:
            ends_at = isoparse(_ends_at)

        serverless_billing_override_definition = cls(
            user_id=user_id,
            machine_type=machine_type,
            unit_price=unit_price,
            starts_at=starts_at,
            is_draft=is_draft,
            ends_at=ends_at,
        )

        serverless_billing_override_definition.additional_properties = d
        return serverless_billing_override_definition

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

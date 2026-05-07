from typing import Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="BillingPaymentFailedPayload")


@_attrs_define
class BillingPaymentFailedPayload:
    """
    Attributes:
        orb_timestamp (Union[Unset, str]):
    """

    orb_timestamp: Union[Unset, str] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        orb_timestamp = self.orb_timestamp

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if orb_timestamp is not UNSET:
            field_dict["orb_timestamp"] = orb_timestamp

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        orb_timestamp = d.pop("orb_timestamp", UNSET)

        billing_payment_failed_payload = cls(
            orb_timestamp=orb_timestamp,
        )

        billing_payment_failed_payload.additional_properties = d
        return billing_payment_failed_payload

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

from typing import Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="CostResult")


@_attrs_define
class CostResult:
    """
    Attributes:
        request_id (str):
        cost_nano_usd (float):
        billable_units (Union[Unset, float]):
        billable_unit (Union[Unset, str]):
    """

    request_id: str
    cost_nano_usd: float
    billable_units: Union[Unset, float] = UNSET
    billable_unit: Union[Unset, str] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        request_id = self.request_id

        cost_nano_usd = self.cost_nano_usd

        billable_units = self.billable_units

        billable_unit = self.billable_unit

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "requestId": request_id,
                "costNanoUsd": cost_nano_usd,
            }
        )
        if billable_units is not UNSET:
            field_dict["billableUnits"] = billable_units
        if billable_unit is not UNSET:
            field_dict["billableUnit"] = billable_unit

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        request_id = d.pop("requestId")

        cost_nano_usd = d.pop("costNanoUsd")

        billable_units = d.pop("billableUnits", UNSET)

        billable_unit = d.pop("billableUnit", UNSET)

        cost_result = cls(
            request_id=request_id,
            cost_nano_usd=cost_nano_usd,
            billable_units=billable_units,
            billable_unit=billable_unit,
        )

        cost_result.additional_properties = d
        return cost_result

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

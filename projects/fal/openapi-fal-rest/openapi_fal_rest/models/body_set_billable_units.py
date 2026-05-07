from typing import Any, TypeVar, Union, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

T = TypeVar("T", bound="BodySetBillableUnits")


@_attrs_define
class BodySetBillableUnits:
    """
    Attributes:
        billable_units (Union[float, str]):
    """

    billable_units: Union[float, str]
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        billable_units: Union[float, str]
        billable_units = self.billable_units

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "billable_units": billable_units,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()

        def _parse_billable_units(data: object) -> Union[float, str]:
            return cast(Union[float, str], data)

        billable_units = _parse_billable_units(d.pop("billable_units"))

        body_set_billable_units = cls(
            billable_units=billable_units,
        )

        body_set_billable_units.additional_properties = d
        return body_set_billable_units

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

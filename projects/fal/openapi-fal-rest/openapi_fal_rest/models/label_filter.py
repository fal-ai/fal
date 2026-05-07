from typing import Any, TypeVar, Union, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.condition_type import ConditionType
from ..types import UNSET, Unset

T = TypeVar("T", bound="LabelFilter")


@_attrs_define
class LabelFilter:
    """Filter for log labels with a specific condition type.

    Attributes:
        key (str):
        value (Union[list[str], str]):
        condition_type (Union[Unset, ConditionType]): Type of condition to apply when filtering logs by label.
    """

    key: str
    value: Union[list[str], str]
    condition_type: Union[Unset, ConditionType] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        key = self.key

        value: Union[list[str], str]
        if isinstance(self.value, list):
            value = self.value

        else:
            value = self.value

        condition_type: Union[Unset, str] = UNSET
        if not isinstance(self.condition_type, Unset):
            condition_type = self.condition_type.value

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "key": key,
                "value": value,
            }
        )
        if condition_type is not UNSET:
            field_dict["condition_type"] = condition_type

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        key = d.pop("key")

        def _parse_value(data: object) -> Union[list[str], str]:
            try:
                if not isinstance(data, list):
                    raise TypeError()
                value_type_1 = cast(list[str], data)

                return value_type_1
            except:  # noqa: E722
                pass
            return cast(Union[list[str], str], data)

        value = _parse_value(d.pop("value"))

        _condition_type = d.pop("condition_type", UNSET)
        condition_type: Union[Unset, ConditionType]
        if isinstance(_condition_type, Unset):
            condition_type = UNSET
        else:
            condition_type = ConditionType(_condition_type)

        label_filter = cls(
            key=key,
            value=value,
            condition_type=condition_type,
        )

        label_filter.additional_properties = d
        return label_filter

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

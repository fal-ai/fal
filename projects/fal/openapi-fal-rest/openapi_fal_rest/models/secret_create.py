from typing import Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="SecretCreate")


@_attrs_define
class SecretCreate:
    """
    Attributes:
        name (str):
        value (str):
        environment_name (Union[Unset, str]):  Default: 'main'.
    """

    name: str
    value: str
    environment_name: Union[Unset, str] = "main"
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        name = self.name

        value = self.value

        environment_name = self.environment_name

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "name": name,
                "value": value,
            }
        )
        if environment_name is not UNSET:
            field_dict["environment_name"] = environment_name

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        name = d.pop("name")

        value = d.pop("value")

        environment_name = d.pop("environment_name", UNSET)

        secret_create = cls(
            name=name,
            value=value,
            environment_name=environment_name,
        )

        secret_create.additional_properties = d
        return secret_create

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

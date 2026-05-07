import datetime
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from dateutil.parser import isoparse

T = TypeVar("T", bound="SecretInfo")


@_attrs_define
class SecretInfo:
    """
    Attributes:
        name (str):
        environment_name (str):
        created_at (datetime.datetime):
        value_length (int):
    """

    name: str
    environment_name: str
    created_at: datetime.datetime
    value_length: int
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        name = self.name

        environment_name = self.environment_name

        created_at = self.created_at.isoformat()

        value_length = self.value_length

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "name": name,
                "environment_name": environment_name,
                "created_at": created_at,
                "value_length": value_length,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        name = d.pop("name")

        environment_name = d.pop("environment_name")

        created_at = isoparse(d.pop("created_at"))

        value_length = d.pop("value_length")

        secret_info = cls(
            name=name,
            environment_name=environment_name,
            created_at=created_at,
            value_length=value_length,
        )

        secret_info.additional_properties = d
        return secret_info

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

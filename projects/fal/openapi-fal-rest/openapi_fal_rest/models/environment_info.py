import datetime
from typing import Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from dateutil.parser import isoparse

from ..types import UNSET, Unset

T = TypeVar("T", bound="EnvironmentInfo")


@_attrs_define
class EnvironmentInfo:
    """Environment information returned by the API.

    Attributes:
        name (str):
        created_at (datetime.datetime):
        description (Union[Unset, str]):
        is_default (Union[Unset, bool]):  Default: False.
    """

    name: str
    created_at: datetime.datetime
    description: Union[Unset, str] = UNSET
    is_default: Union[Unset, bool] = False
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        name = self.name

        created_at = self.created_at.isoformat()

        description = self.description

        is_default = self.is_default

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "name": name,
                "created_at": created_at,
            }
        )
        if description is not UNSET:
            field_dict["description"] = description
        if is_default is not UNSET:
            field_dict["is_default"] = is_default

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        name = d.pop("name")

        created_at = isoparse(d.pop("created_at"))

        description = d.pop("description", UNSET)

        is_default = d.pop("is_default", UNSET)

        environment_info = cls(
            name=name,
            created_at=created_at,
            description=description,
            is_default=is_default,
        )

        environment_info.additional_properties = d
        return environment_info

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

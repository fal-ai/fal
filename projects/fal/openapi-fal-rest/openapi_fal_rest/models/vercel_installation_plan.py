from typing import Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="VercelInstallationPlan")


@_attrs_define
class VercelInstallationPlan:
    """
    Attributes:
        id (Union[Unset, str]):  Default: 'default'.
        name (Union[Unset, str]):  Default: 'Fal Pre-paid Plan'.
        description (Union[Unset, str]):  Default: 'Use your credits to run any model that is available on fal'.
        scope (Union[Unset, str]):  Default: 'installation'.
        type_ (Union[Unset, str]):  Default: 'prepayment'.
    """

    id: Union[Unset, str] = "default"
    name: Union[Unset, str] = "Fal Pre-paid Plan"
    description: Union[Unset, str] = "Use your credits to run any model that is available on fal"
    scope: Union[Unset, str] = "installation"
    type_: Union[Unset, str] = "prepayment"
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        id = self.id

        name = self.name

        description = self.description

        scope = self.scope

        type_ = self.type_

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if id is not UNSET:
            field_dict["id"] = id
        if name is not UNSET:
            field_dict["name"] = name
        if description is not UNSET:
            field_dict["description"] = description
        if scope is not UNSET:
            field_dict["scope"] = scope
        if type_ is not UNSET:
            field_dict["type"] = type_

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        id = d.pop("id", UNSET)

        name = d.pop("name", UNSET)

        description = d.pop("description", UNSET)

        scope = d.pop("scope", UNSET)

        type_ = d.pop("type", UNSET)

        vercel_installation_plan = cls(
            id=id,
            name=name,
            description=description,
            scope=scope,
            type_=type_,
        )

        vercel_installation_plan.additional_properties = d
        return vercel_installation_plan

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

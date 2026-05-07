from typing import TYPE_CHECKING, Any, TypeVar, Union, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.key_policy_restrictions import KeyPolicyRestrictions


T = TypeVar("T", bound="KeyPolicy")


@_attrs_define
class KeyPolicy:
    """
    Attributes:
        permissions (list[str]):
        restrictions (Union[Unset, KeyPolicyRestrictions]):
    """

    permissions: list[str]
    restrictions: Union[Unset, "KeyPolicyRestrictions"] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        permissions = self.permissions

        restrictions: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.restrictions, Unset):
            restrictions = self.restrictions.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "permissions": permissions,
            }
        )
        if restrictions is not UNSET:
            field_dict["restrictions"] = restrictions

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        from ..models.key_policy_restrictions import KeyPolicyRestrictions

        d = src_dict.copy()
        permissions = cast(list[str], d.pop("permissions"))

        _restrictions = d.pop("restrictions", UNSET)
        restrictions: Union[Unset, KeyPolicyRestrictions]
        if isinstance(_restrictions, Unset):
            restrictions = UNSET
        else:
            restrictions = KeyPolicyRestrictions.from_dict(_restrictions)

        key_policy = cls(
            permissions=permissions,
            restrictions=restrictions,
        )

        key_policy.additional_properties = d
        return key_policy

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

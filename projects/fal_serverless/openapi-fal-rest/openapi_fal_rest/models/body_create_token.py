from typing import Any, Dict, List, Type, TypeVar, Union, cast

import attr

from ..types import UNSET, Unset

T = TypeVar("T", bound="BodyCreateToken")


@attr.s(auto_attribs=True)
class BodyCreateToken:
    """
    Attributes:
        allowed_apps (List[str]):
        token_expiration (Union[Unset, int]):  Default: 300.
    """

    allowed_apps: List[str]
    token_expiration: Union[Unset, int] = 300
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        allowed_apps = self.allowed_apps

        token_expiration = self.token_expiration

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "allowed_apps": allowed_apps,
            }
        )
        if token_expiration is not UNSET:
            field_dict["token_expiration"] = token_expiration

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        allowed_apps = cast(List[str], d.pop("allowed_apps"))

        token_expiration = d.pop("token_expiration", UNSET)

        body_create_token = cls(
            allowed_apps=allowed_apps,
            token_expiration=token_expiration,
        )

        body_create_token.additional_properties = d
        return body_create_token

    @property
    def additional_keys(self) -> List[str]:
        return list(self.additional_properties.keys())

    def __getitem__(self, key: str) -> Any:
        return self.additional_properties[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.additional_properties[key] = value

    def __delitem__(self, key: str) -> None:
        del self.additional_properties[key]

    def __contains__(self, key: str) -> bool:
        return key in self.additional_properties

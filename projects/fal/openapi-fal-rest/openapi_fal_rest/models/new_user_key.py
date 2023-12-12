from typing import Any, Dict, List, Type, TypeVar

import attr

T = TypeVar("T", bound="NewUserKey")


@attr.s(auto_attribs=True)
class NewUserKey:
    """
    Attributes:
        key_id (str):
        key_secret (str):
    """

    key_id: str
    key_secret: str
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        key_id = self.key_id
        key_secret = self.key_secret

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "key_id": key_id,
                "key_secret": key_secret,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        key_id = d.pop("key_id")

        key_secret = d.pop("key_secret")

        new_user_key = cls(
            key_id=key_id,
            key_secret=key_secret,
        )

        new_user_key.additional_properties = d
        return new_user_key

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

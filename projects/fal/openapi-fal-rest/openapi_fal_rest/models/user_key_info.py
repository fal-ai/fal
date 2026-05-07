import datetime
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from dateutil.parser import isoparse

from ..models.key_scope import KeyScope

T = TypeVar("T", bound="UserKeyInfo")


@_attrs_define
class UserKeyInfo:
    """
    Attributes:
        user_id (str):
        key_id (str):
        created_at (datetime.datetime):
        scope (KeyScope):
        alias (str):
    """

    user_id: str
    key_id: str
    created_at: datetime.datetime
    scope: KeyScope
    alias: str
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        user_id = self.user_id

        key_id = self.key_id

        created_at = self.created_at.isoformat()

        scope = self.scope.value

        alias = self.alias

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "user_id": user_id,
                "key_id": key_id,
                "created_at": created_at,
                "scope": scope,
                "alias": alias,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        user_id = d.pop("user_id")

        key_id = d.pop("key_id")

        created_at = isoparse(d.pop("created_at"))

        scope = KeyScope(d.pop("scope"))

        alias = d.pop("alias")

        user_key_info = cls(
            user_id=user_id,
            key_id=key_id,
            created_at=created_at,
            scope=scope,
            alias=alias,
        )

        user_key_info.additional_properties = d
        return user_key_info

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

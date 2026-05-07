from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

T = TypeVar("T", bound="EnrichedKeyInfo")


@_attrs_define
class EnrichedKeyInfo:
    """Database information about an API key for invoice enrichment.

    Attributes:
        key_id (str):
        alias (str):
        user_id (str):
        nickname (str):
        email (str):
        scope (str):
        created_at (str):
    """

    key_id: str
    alias: str
    user_id: str
    nickname: str
    email: str
    scope: str
    created_at: str
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        key_id = self.key_id

        alias = self.alias

        user_id = self.user_id

        nickname = self.nickname

        email = self.email

        scope = self.scope

        created_at = self.created_at

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "key_id": key_id,
                "alias": alias,
                "user_id": user_id,
                "nickname": nickname,
                "email": email,
                "scope": scope,
                "created_at": created_at,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        key_id = d.pop("key_id")

        alias = d.pop("alias")

        user_id = d.pop("user_id")

        nickname = d.pop("nickname")

        email = d.pop("email")

        scope = d.pop("scope")

        created_at = d.pop("created_at")

        enriched_key_info = cls(
            key_id=key_id,
            alias=alias,
            user_id=user_id,
            nickname=nickname,
            email=email,
            scope=scope,
            created_at=created_at,
        )

        enriched_key_info.additional_properties = d
        return enriched_key_info

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

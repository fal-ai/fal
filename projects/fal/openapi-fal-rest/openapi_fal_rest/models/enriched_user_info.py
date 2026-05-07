from typing import Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.enriched_user_info_lookup_method import EnrichedUserInfoLookupMethod
from ..types import UNSET, Unset

T = TypeVar("T", bound="EnrichedUserInfo")


@_attrs_define
class EnrichedUserInfo:
    """Database information about a playground user for invoice enrichment.

    Attributes:
        user_id (str):
        nickname (str):
        email (str):
        full_name (str):
        personal_auth_id (str):
        lookup_method (EnrichedUserInfoLookupMethod):
        workos_sub (Union[Unset, str]):
    """

    user_id: str
    nickname: str
    email: str
    full_name: str
    personal_auth_id: str
    lookup_method: EnrichedUserInfoLookupMethod
    workos_sub: Union[Unset, str] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        user_id = self.user_id

        nickname = self.nickname

        email = self.email

        full_name = self.full_name

        personal_auth_id = self.personal_auth_id

        lookup_method = self.lookup_method.value

        workos_sub = self.workos_sub

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "user_id": user_id,
                "nickname": nickname,
                "email": email,
                "full_name": full_name,
                "personal_auth_id": personal_auth_id,
                "lookup_method": lookup_method,
            }
        )
        if workos_sub is not UNSET:
            field_dict["workos_sub"] = workos_sub

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        user_id = d.pop("user_id")

        nickname = d.pop("nickname")

        email = d.pop("email")

        full_name = d.pop("full_name")

        personal_auth_id = d.pop("personal_auth_id")

        lookup_method = EnrichedUserInfoLookupMethod(d.pop("lookup_method"))

        workos_sub = d.pop("workos_sub", UNSET)

        enriched_user_info = cls(
            user_id=user_id,
            nickname=nickname,
            email=email,
            full_name=full_name,
            personal_auth_id=personal_auth_id,
            lookup_method=lookup_method,
            workos_sub=workos_sub,
        )

        enriched_user_info.additional_properties = d
        return enriched_user_info

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

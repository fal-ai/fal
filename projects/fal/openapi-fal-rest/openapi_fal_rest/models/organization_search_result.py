from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

T = TypeVar("T", bound="OrganizationSearchResult")


@_attrs_define
class OrganizationSearchResult:
    """Search result for an organization.

    Attributes:
        org_user_id (str):
        nickname (str):
        full_name (str):
        email (str):
    """

    org_user_id: str
    nickname: str
    full_name: str
    email: str
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        org_user_id = self.org_user_id

        nickname = self.nickname

        full_name = self.full_name

        email = self.email

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "org_user_id": org_user_id,
                "nickname": nickname,
                "full_name": full_name,
                "email": email,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        org_user_id = d.pop("org_user_id")

        nickname = d.pop("nickname")

        full_name = d.pop("full_name")

        email = d.pop("email")

        organization_search_result = cls(
            org_user_id=org_user_id,
            nickname=nickname,
            full_name=full_name,
            email=email,
        )

        organization_search_result.additional_properties = d
        return organization_search_result

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

import datetime
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from dateutil.parser import isoparse

T = TypeVar("T", bound="CreateOrganizationResponse")


@_attrs_define
class CreateOrganizationResponse:
    """Response after creating an organization.

    Attributes:
        org_user_id (str):
        nickname (str):
        full_name (str):
        email (str):
        created_at (datetime.datetime):
    """

    org_user_id: str
    nickname: str
    full_name: str
    email: str
    created_at: datetime.datetime
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        org_user_id = self.org_user_id

        nickname = self.nickname

        full_name = self.full_name

        email = self.email

        created_at = self.created_at.isoformat()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "org_user_id": org_user_id,
                "nickname": nickname,
                "full_name": full_name,
                "email": email,
                "created_at": created_at,
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

        created_at = isoparse(d.pop("created_at"))

        create_organization_response = cls(
            org_user_id=org_user_id,
            nickname=nickname,
            full_name=full_name,
            email=email,
            created_at=created_at,
        )

        create_organization_response.additional_properties = d
        return create_organization_response

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

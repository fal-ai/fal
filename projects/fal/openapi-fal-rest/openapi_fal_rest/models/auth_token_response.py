from typing import Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="AuthTokenResponse")


@_attrs_define
class AuthTokenResponse:
    """
    Attributes:
        token (str):
        created_at (str):
        expires_at (str):
        token_type (Union[Unset, str]):  Default: 'Bearer'.
        base_url (Union[Unset, str]):  Default: 'https://v2.fal.media'.
    """

    token: str
    created_at: str
    expires_at: str
    token_type: Union[Unset, str] = "Bearer"
    base_url: Union[Unset, str] = "https://v2.fal.media"
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        token = self.token

        created_at = self.created_at

        expires_at = self.expires_at

        token_type = self.token_type

        base_url = self.base_url

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "token": token,
                "created_at": created_at,
                "expires_at": expires_at,
            }
        )
        if token_type is not UNSET:
            field_dict["token_type"] = token_type
        if base_url is not UNSET:
            field_dict["base_url"] = base_url

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        token = d.pop("token")

        created_at = d.pop("created_at")

        expires_at = d.pop("expires_at")

        token_type = d.pop("token_type", UNSET)

        base_url = d.pop("base_url", UNSET)

        auth_token_response = cls(
            token=token,
            created_at=created_at,
            expires_at=expires_at,
            token_type=token_type,
            base_url=base_url,
        )

        auth_token_response.additional_properties = d
        return auth_token_response

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

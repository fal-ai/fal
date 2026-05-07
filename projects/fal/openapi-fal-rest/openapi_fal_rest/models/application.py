from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.application_auth_mode import ApplicationAuthMode

T = TypeVar("T", bound="Application")


@_attrs_define
class Application:
    """
    Attributes:
        user (str):
        name (str):
        auth (ApplicationAuthMode):
    """

    user: str
    name: str
    auth: ApplicationAuthMode
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        user = self.user

        name = self.name

        auth = self.auth.value

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "user": user,
                "name": name,
                "auth": auth,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        user = d.pop("user")

        name = d.pop("name")

        auth = ApplicationAuthMode(d.pop("auth"))

        application = cls(
            user=user,
            name=name,
            auth=auth,
        )

        application.additional_properties = d
        return application

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

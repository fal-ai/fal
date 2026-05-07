from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

if TYPE_CHECKING:
    from ..models.vercel_credentials import VercelCredentials


T = TypeVar("T", bound="UpsertInstallationInput")


@_attrs_define
class UpsertInstallationInput:
    """
    Attributes:
        credentials (VercelCredentials):
    """

    credentials: "VercelCredentials"
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        credentials = self.credentials.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "credentials": credentials,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        from ..models.vercel_credentials import VercelCredentials

        d = src_dict.copy()
        credentials = VercelCredentials.from_dict(d.pop("credentials"))

        upsert_installation_input = cls(
            credentials=credentials,
        )

        upsert_installation_input.additional_properties = d
        return upsert_installation_input

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

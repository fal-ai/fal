from typing import TYPE_CHECKING, Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.resource_secret import ResourceSecret


T = TypeVar("T", bound="RotateSecretsResponse")


@_attrs_define
class RotateSecretsResponse:
    """
    Attributes:
        sync (bool):
        secrets (Union[Unset, list['ResourceSecret']]):
        partial (Union[Unset, bool]):  Default: False.
    """

    sync: bool
    secrets: Union[Unset, list["ResourceSecret"]] = UNSET
    partial: Union[Unset, bool] = False
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        sync = self.sync

        secrets: Union[Unset, list[dict[str, Any]]] = UNSET
        if not isinstance(self.secrets, Unset):
            secrets = []
            for secrets_item_data in self.secrets:
                secrets_item = secrets_item_data.to_dict()
                secrets.append(secrets_item)

        partial = self.partial

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "sync": sync,
            }
        )
        if secrets is not UNSET:
            field_dict["secrets"] = secrets
        if partial is not UNSET:
            field_dict["partial"] = partial

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        from ..models.resource_secret import ResourceSecret

        d = src_dict.copy()
        sync = d.pop("sync")

        secrets = []
        _secrets = d.pop("secrets", UNSET)
        for secrets_item_data in _secrets or []:
            secrets_item = ResourceSecret.from_dict(secrets_item_data)

            secrets.append(secrets_item)

        partial = d.pop("partial", UNSET)

        rotate_secrets_response = cls(
            sync=sync,
            secrets=secrets,
            partial=partial,
        )

        rotate_secrets_response.additional_properties = d
        return rotate_secrets_response

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

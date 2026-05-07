from typing import TYPE_CHECKING, Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.batch_invite_item import BatchInviteItem


T = TypeVar("T", bound="BatchInviteRequest")


@_attrs_define
class BatchInviteRequest:
    """
    Attributes:
        invites (list['BatchInviteItem']):
        required_sso_connection (Union[Unset, str]):
    """

    invites: list["BatchInviteItem"]
    required_sso_connection: Union[Unset, str] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        invites = []
        for invites_item_data in self.invites:
            invites_item = invites_item_data.to_dict()
            invites.append(invites_item)

        required_sso_connection = self.required_sso_connection

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "invites": invites,
            }
        )
        if required_sso_connection is not UNSET:
            field_dict["required_sso_connection"] = required_sso_connection

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        from ..models.batch_invite_item import BatchInviteItem

        d = src_dict.copy()
        invites = []
        _invites = d.pop("invites")
        for invites_item_data in _invites:
            invites_item = BatchInviteItem.from_dict(invites_item_data)

            invites.append(invites_item)

        required_sso_connection = d.pop("required_sso_connection", UNSET)

        batch_invite_request = cls(
            invites=invites,
            required_sso_connection=required_sso_connection,
        )

        batch_invite_request.additional_properties = d
        return batch_invite_request

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

from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

if TYPE_CHECKING:
    from ..models.playground_user_entry import PlaygroundUserEntry


T = TypeVar("T", bound="PlaygroundSubsection")


@_attrs_define
class PlaygroundSubsection:
    """Playground subsection with user breakdown.

    Attributes:
        type_ (str):
        total_amount (str):
        users (list['PlaygroundUserEntry']):
    """

    type_: str
    total_amount: str
    users: list["PlaygroundUserEntry"]
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        type_ = self.type_

        total_amount = self.total_amount

        users = []
        for users_item_data in self.users:
            users_item = users_item_data.to_dict()
            users.append(users_item)

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "type": type_,
                "total_amount": total_amount,
                "users": users,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        from ..models.playground_user_entry import PlaygroundUserEntry

        d = src_dict.copy()
        type_ = d.pop("type")

        total_amount = d.pop("total_amount")

        users = []
        _users = d.pop("users")
        for users_item_data in _users:
            users_item = PlaygroundUserEntry.from_dict(users_item_data)

            users.append(users_item)

        playground_subsection = cls(
            type_=type_,
            total_amount=total_amount,
            users=users,
        )

        playground_subsection.additional_properties = d
        return playground_subsection

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

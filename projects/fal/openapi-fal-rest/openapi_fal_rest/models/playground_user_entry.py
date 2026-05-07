from typing import TYPE_CHECKING, Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.playground_user_entry_auth_provider import PlaygroundUserEntryAuthProvider
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.enriched_user_info import EnrichedUserInfo
    from ..models.line_item import LineItem


T = TypeVar("T", bound="PlaygroundUserEntry")


@_attrs_define
class PlaygroundUserEntry:
    """Playground usage for a single user.

    Attributes:
        user_id (str):
        auth_provider (PlaygroundUserEntryAuthProvider):
        total_amount (str):
        items (list['LineItem']):
        user_info (Union[Unset, EnrichedUserInfo]): Database information about a playground user for invoice enrichment.
        total_platform_discount_amount (Union[Unset, float]):  Default: 0.0.
        total_endpoint_discount_amount (Union[Unset, float]):  Default: 0.0.
    """

    user_id: str
    auth_provider: PlaygroundUserEntryAuthProvider
    total_amount: str
    items: list["LineItem"]
    user_info: Union[Unset, "EnrichedUserInfo"] = UNSET
    total_platform_discount_amount: Union[Unset, float] = 0.0
    total_endpoint_discount_amount: Union[Unset, float] = 0.0
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        user_id = self.user_id

        auth_provider = self.auth_provider.value

        total_amount = self.total_amount

        items = []
        for items_item_data in self.items:
            items_item = items_item_data.to_dict()
            items.append(items_item)

        user_info: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.user_info, Unset):
            user_info = self.user_info.to_dict()

        total_platform_discount_amount = self.total_platform_discount_amount

        total_endpoint_discount_amount = self.total_endpoint_discount_amount

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "user_id": user_id,
                "auth_provider": auth_provider,
                "total_amount": total_amount,
                "items": items,
            }
        )
        if user_info is not UNSET:
            field_dict["user_info"] = user_info
        if total_platform_discount_amount is not UNSET:
            field_dict["total_platform_discount_amount"] = total_platform_discount_amount
        if total_endpoint_discount_amount is not UNSET:
            field_dict["total_endpoint_discount_amount"] = total_endpoint_discount_amount

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        from ..models.enriched_user_info import EnrichedUserInfo
        from ..models.line_item import LineItem

        d = src_dict.copy()
        user_id = d.pop("user_id")

        auth_provider = PlaygroundUserEntryAuthProvider(d.pop("auth_provider"))

        total_amount = d.pop("total_amount")

        items = []
        _items = d.pop("items")
        for items_item_data in _items:
            items_item = LineItem.from_dict(items_item_data)

            items.append(items_item)

        _user_info = d.pop("user_info", UNSET)
        user_info: Union[Unset, EnrichedUserInfo]
        if isinstance(_user_info, Unset):
            user_info = UNSET
        else:
            user_info = EnrichedUserInfo.from_dict(_user_info)

        total_platform_discount_amount = d.pop("total_platform_discount_amount", UNSET)

        total_endpoint_discount_amount = d.pop("total_endpoint_discount_amount", UNSET)

        playground_user_entry = cls(
            user_id=user_id,
            auth_provider=auth_provider,
            total_amount=total_amount,
            items=items,
            user_info=user_info,
            total_platform_discount_amount=total_platform_discount_amount,
            total_endpoint_discount_amount=total_endpoint_discount_amount,
        )

        playground_user_entry.additional_properties = d
        return playground_user_entry

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

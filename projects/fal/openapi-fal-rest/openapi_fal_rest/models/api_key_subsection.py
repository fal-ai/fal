from typing import TYPE_CHECKING, Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.enriched_key_info import EnrichedKeyInfo
    from ..models.line_item import LineItem


T = TypeVar("T", bound="ApiKeySubsection")


@_attrs_define
class ApiKeySubsection:
    """API key subsection with usage breakdown.

    Attributes:
        type_ (str):
        key_id (str):
        total_amount (str):
        total_quantity (float):
        items (list['LineItem']):
        key_info (Union[Unset, EnrichedKeyInfo]): Database information about an API key for invoice enrichment.
        total_platform_discount_amount (Union[Unset, float]):  Default: 0.0.
        total_endpoint_discount_amount (Union[Unset, float]):  Default: 0.0.
    """

    type_: str
    key_id: str
    total_amount: str
    total_quantity: float
    items: list["LineItem"]
    key_info: Union[Unset, "EnrichedKeyInfo"] = UNSET
    total_platform_discount_amount: Union[Unset, float] = 0.0
    total_endpoint_discount_amount: Union[Unset, float] = 0.0
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        type_ = self.type_

        key_id = self.key_id

        total_amount = self.total_amount

        total_quantity = self.total_quantity

        items = []
        for items_item_data in self.items:
            items_item = items_item_data.to_dict()
            items.append(items_item)

        key_info: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.key_info, Unset):
            key_info = self.key_info.to_dict()

        total_platform_discount_amount = self.total_platform_discount_amount

        total_endpoint_discount_amount = self.total_endpoint_discount_amount

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "type": type_,
                "key_id": key_id,
                "total_amount": total_amount,
                "total_quantity": total_quantity,
                "items": items,
            }
        )
        if key_info is not UNSET:
            field_dict["key_info"] = key_info
        if total_platform_discount_amount is not UNSET:
            field_dict["total_platform_discount_amount"] = total_platform_discount_amount
        if total_endpoint_discount_amount is not UNSET:
            field_dict["total_endpoint_discount_amount"] = total_endpoint_discount_amount

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        from ..models.enriched_key_info import EnrichedKeyInfo
        from ..models.line_item import LineItem

        d = src_dict.copy()
        type_ = d.pop("type")

        key_id = d.pop("key_id")

        total_amount = d.pop("total_amount")

        total_quantity = d.pop("total_quantity")

        items = []
        _items = d.pop("items")
        for items_item_data in _items:
            items_item = LineItem.from_dict(items_item_data)

            items.append(items_item)

        _key_info = d.pop("key_info", UNSET)
        key_info: Union[Unset, EnrichedKeyInfo]
        if isinstance(_key_info, Unset):
            key_info = UNSET
        else:
            key_info = EnrichedKeyInfo.from_dict(_key_info)

        total_platform_discount_amount = d.pop("total_platform_discount_amount", UNSET)

        total_endpoint_discount_amount = d.pop("total_endpoint_discount_amount", UNSET)

        api_key_subsection = cls(
            type_=type_,
            key_id=key_id,
            total_amount=total_amount,
            total_quantity=total_quantity,
            items=items,
            key_info=key_info,
            total_platform_discount_amount=total_platform_discount_amount,
            total_endpoint_discount_amount=total_endpoint_discount_amount,
        )

        api_key_subsection.additional_properties = d
        return api_key_subsection

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

from typing import TYPE_CHECKING, Any, Dict, List, Type, TypeVar, Union

import attr

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.comfy_workflow_item import ComfyWorkflowItem


T = TypeVar("T", bound="PageComfyWorkflowItem")


@attr.s(auto_attribs=True)
class PageComfyWorkflowItem:
    """
    Attributes:
        items (List['ComfyWorkflowItem']):
        total (Union[Unset, int]):
        page (Union[Unset, int]):
        size (Union[Unset, int]):
        pages (Union[Unset, int]):
    """

    items: List["ComfyWorkflowItem"]
    total: Union[Unset, int] = UNSET
    page: Union[Unset, int] = UNSET
    size: Union[Unset, int] = UNSET
    pages: Union[Unset, int] = UNSET
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        items = []
        for items_item_data in self.items:
            items_item = items_item_data.to_dict()

            items.append(items_item)

        total = self.total
        page = self.page
        size = self.size
        pages = self.pages

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "items": items,
            }
        )
        if total is not UNSET:
            field_dict["total"] = total
        if page is not UNSET:
            field_dict["page"] = page
        if size is not UNSET:
            field_dict["size"] = size
        if pages is not UNSET:
            field_dict["pages"] = pages

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        from ..models.comfy_workflow_item import ComfyWorkflowItem

        d = src_dict.copy()
        items = []
        _items = d.pop("items")
        for items_item_data in _items:
            items_item = ComfyWorkflowItem.from_dict(items_item_data)

            items.append(items_item)

        total = d.pop("total", UNSET)

        page = d.pop("page", UNSET)

        size = d.pop("size", UNSET)

        pages = d.pop("pages", UNSET)

        page_comfy_workflow_item = cls(
            items=items,
            total=total,
            page=page,
            size=size,
            pages=pages,
        )

        page_comfy_workflow_item.additional_properties = d
        return page_comfy_workflow_item

    @property
    def additional_keys(self) -> List[str]:
        return list(self.additional_properties.keys())

    def __getitem__(self, key: str) -> Any:
        return self.additional_properties[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.additional_properties[key] = value

    def __delitem__(self, key: str) -> None:
        del self.additional_properties[key]

    def __contains__(self, key: str) -> bool:
        return key in self.additional_properties

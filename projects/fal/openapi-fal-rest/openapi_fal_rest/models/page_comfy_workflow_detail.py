from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

if TYPE_CHECKING:
    from ..models.comfy_workflow_detail import ComfyWorkflowDetail


T = TypeVar("T", bound="PageComfyWorkflowDetail")


@_attrs_define
class PageComfyWorkflowDetail:
    """
    Attributes:
        items (list['ComfyWorkflowDetail']):
        total (int):
        page (int):
        size (int):
        pages (int):
    """

    items: list["ComfyWorkflowDetail"]
    total: int
    page: int
    size: int
    pages: int
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        items = []
        for items_item_data in self.items:
            items_item = items_item_data.to_dict()
            items.append(items_item)

        total = self.total

        page = self.page

        size = self.size

        pages = self.pages

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "items": items,
                "total": total,
                "page": page,
                "size": size,
                "pages": pages,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        from ..models.comfy_workflow_detail import ComfyWorkflowDetail

        d = src_dict.copy()
        items = []
        _items = d.pop("items")
        for items_item_data in _items:
            items_item = ComfyWorkflowDetail.from_dict(items_item_data)

            items.append(items_item)

        total = d.pop("total")

        page = d.pop("page")

        size = d.pop("size")

        pages = d.pop("pages")

        page_comfy_workflow_detail = cls(
            items=items,
            total=total,
            page=page,
            size=size,
            pages=pages,
        )

        page_comfy_workflow_detail.additional_properties = d
        return page_comfy_workflow_detail

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

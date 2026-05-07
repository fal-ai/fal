from typing import TYPE_CHECKING, Any, TypeVar, Union, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.document_ingested_item import DocumentIngestedItem


T = TypeVar("T", bound="DocumentIngestResponse")


@_attrs_define
class DocumentIngestResponse:
    """
    Attributes:
        request_id (Union[Unset, str]):
        vector_ids (Union[Unset, list[str]]):
        items (Union[Unset, list['DocumentIngestedItem']]):
    """

    request_id: Union[Unset, str] = UNSET
    vector_ids: Union[Unset, list[str]] = UNSET
    items: Union[Unset, list["DocumentIngestedItem"]] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        request_id = self.request_id

        vector_ids: Union[Unset, list[str]] = UNSET
        if not isinstance(self.vector_ids, Unset):
            vector_ids = self.vector_ids

        items: Union[Unset, list[dict[str, Any]]] = UNSET
        if not isinstance(self.items, Unset):
            items = []
            for items_item_data in self.items:
                items_item = items_item_data.to_dict()
                items.append(items_item)

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if request_id is not UNSET:
            field_dict["request_id"] = request_id
        if vector_ids is not UNSET:
            field_dict["vector_ids"] = vector_ids
        if items is not UNSET:
            field_dict["items"] = items

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        from ..models.document_ingested_item import DocumentIngestedItem

        d = src_dict.copy()
        request_id = d.pop("request_id", UNSET)

        vector_ids = cast(list[str], d.pop("vector_ids", UNSET))

        items = []
        _items = d.pop("items", UNSET)
        for items_item_data in _items or []:
            items_item = DocumentIngestedItem.from_dict(items_item_data)

            items.append(items_item)

        document_ingest_response = cls(
            request_id=request_id,
            vector_ids=vector_ids,
            items=items,
        )

        document_ingest_response.additional_properties = d
        return document_ingest_response

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

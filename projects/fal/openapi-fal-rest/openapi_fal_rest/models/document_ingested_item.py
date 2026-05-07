from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

T = TypeVar("T", bound="DocumentIngestedItem")


@_attrs_define
class DocumentIngestedItem:
    """
    Attributes:
        vector_id (str):
        item_type (str):
        source (str):
    """

    vector_id: str
    item_type: str
    source: str
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        vector_id = self.vector_id

        item_type = self.item_type

        source = self.source

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "vector_id": vector_id,
                "item_type": item_type,
                "source": source,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        vector_id = d.pop("vector_id")

        item_type = d.pop("item_type")

        source = d.pop("source")

        document_ingested_item = cls(
            vector_id=vector_id,
            item_type=item_type,
            source=source,
        )

        document_ingested_item.additional_properties = d
        return document_ingested_item

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

from typing import TYPE_CHECKING, Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.document_browse_response_documents_item import DocumentBrowseResponseDocumentsItem


T = TypeVar("T", bound="DocumentBrowseResponse")


@_attrs_define
class DocumentBrowseResponse:
    """
    Attributes:
        documents (list['DocumentBrowseResponseDocumentsItem']):
        next_cursor (Union[Unset, str]):
    """

    documents: list["DocumentBrowseResponseDocumentsItem"]
    next_cursor: Union[Unset, str] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        documents = []
        for documents_item_data in self.documents:
            documents_item = documents_item_data.to_dict()
            documents.append(documents_item)

        next_cursor = self.next_cursor

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "documents": documents,
            }
        )
        if next_cursor is not UNSET:
            field_dict["next_cursor"] = next_cursor

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        from ..models.document_browse_response_documents_item import DocumentBrowseResponseDocumentsItem

        d = src_dict.copy()
        documents = []
        _documents = d.pop("documents")
        for documents_item_data in _documents:
            documents_item = DocumentBrowseResponseDocumentsItem.from_dict(documents_item_data)

            documents.append(documents_item)

        next_cursor = d.pop("next_cursor", UNSET)

        document_browse_response = cls(
            documents=documents,
            next_cursor=next_cursor,
        )

        document_browse_response.additional_properties = d
        return document_browse_response

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

from typing import Any, TypeVar, Union
from uuid import UUID

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.document_ingest_request_media_type import DocumentIngestRequestMediaType
from ..types import UNSET, Unset

T = TypeVar("T", bound="DocumentIngestRequest")


@_attrs_define
class DocumentIngestRequest:
    """
    Attributes:
        request_id (Union[Unset, UUID]):
        image_url (Union[Unset, str]):
        media_url (Union[Unset, str]):
        media_type (Union[Unset, DocumentIngestRequestMediaType]):
        prompt (Union[Unset, str]):
    """

    request_id: Union[Unset, UUID] = UNSET
    image_url: Union[Unset, str] = UNSET
    media_url: Union[Unset, str] = UNSET
    media_type: Union[Unset, DocumentIngestRequestMediaType] = UNSET
    prompt: Union[Unset, str] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        request_id: Union[Unset, str] = UNSET
        if not isinstance(self.request_id, Unset):
            request_id = str(self.request_id)

        image_url = self.image_url

        media_url = self.media_url

        media_type: Union[Unset, str] = UNSET
        if not isinstance(self.media_type, Unset):
            media_type = self.media_type.value

        prompt = self.prompt

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if request_id is not UNSET:
            field_dict["request_id"] = request_id
        if image_url is not UNSET:
            field_dict["image_url"] = image_url
        if media_url is not UNSET:
            field_dict["media_url"] = media_url
        if media_type is not UNSET:
            field_dict["media_type"] = media_type
        if prompt is not UNSET:
            field_dict["prompt"] = prompt

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        _request_id = d.pop("request_id", UNSET)
        request_id: Union[Unset, UUID]
        if isinstance(_request_id, Unset):
            request_id = UNSET
        else:
            request_id = UUID(_request_id)

        image_url = d.pop("image_url", UNSET)

        media_url = d.pop("media_url", UNSET)

        _media_type = d.pop("media_type", UNSET)
        media_type: Union[Unset, DocumentIngestRequestMediaType]
        if isinstance(_media_type, Unset):
            media_type = UNSET
        else:
            media_type = DocumentIngestRequestMediaType(_media_type)

        prompt = d.pop("prompt", UNSET)

        document_ingest_request = cls(
            request_id=request_id,
            image_url=image_url,
            media_url=media_url,
            media_type=media_type,
            prompt=prompt,
        )

        document_ingest_request.additional_properties = d
        return document_ingest_request

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

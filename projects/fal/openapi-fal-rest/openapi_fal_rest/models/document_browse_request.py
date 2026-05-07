from typing import Any, TypeVar, Union, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="DocumentBrowseRequest")


@_attrs_define
class DocumentBrowseRequest:
    """
    Attributes:
        top_k (Union[Unset, int]):  Default: 100.
        cursor (Union[Unset, str]):
        include_attributes (Union[Unset, list[str]]):
        filters (Union[Unset, Any]):
        search_text (Union[Unset, str]):
        use_hybrid_text_search (Union[Unset, bool]):  Default: True.
        search_image_url (Union[Unset, str]):
        search_video_url (Union[Unset, str]):
        search_3d_url (Union[Unset, str]):
        search_audio_url (Union[Unset, str]):
    """

    top_k: Union[Unset, int] = 100
    cursor: Union[Unset, str] = UNSET
    include_attributes: Union[Unset, list[str]] = UNSET
    filters: Union[Unset, Any] = UNSET
    search_text: Union[Unset, str] = UNSET
    use_hybrid_text_search: Union[Unset, bool] = True
    search_image_url: Union[Unset, str] = UNSET
    search_video_url: Union[Unset, str] = UNSET
    search_3d_url: Union[Unset, str] = UNSET
    search_audio_url: Union[Unset, str] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        top_k = self.top_k

        cursor = self.cursor

        include_attributes: Union[Unset, list[str]] = UNSET
        if not isinstance(self.include_attributes, Unset):
            include_attributes = self.include_attributes

        filters = self.filters

        search_text = self.search_text

        use_hybrid_text_search = self.use_hybrid_text_search

        search_image_url = self.search_image_url

        search_video_url = self.search_video_url

        search_3d_url = self.search_3d_url

        search_audio_url = self.search_audio_url

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if top_k is not UNSET:
            field_dict["top_k"] = top_k
        if cursor is not UNSET:
            field_dict["cursor"] = cursor
        if include_attributes is not UNSET:
            field_dict["include_attributes"] = include_attributes
        if filters is not UNSET:
            field_dict["filters"] = filters
        if search_text is not UNSET:
            field_dict["search_text"] = search_text
        if use_hybrid_text_search is not UNSET:
            field_dict["use_hybrid_text_search"] = use_hybrid_text_search
        if search_image_url is not UNSET:
            field_dict["search_image_url"] = search_image_url
        if search_video_url is not UNSET:
            field_dict["search_video_url"] = search_video_url
        if search_3d_url is not UNSET:
            field_dict["search_3d_url"] = search_3d_url
        if search_audio_url is not UNSET:
            field_dict["search_audio_url"] = search_audio_url

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        top_k = d.pop("top_k", UNSET)

        cursor = d.pop("cursor", UNSET)

        include_attributes = cast(list[str], d.pop("include_attributes", UNSET))

        filters = d.pop("filters", UNSET)

        search_text = d.pop("search_text", UNSET)

        use_hybrid_text_search = d.pop("use_hybrid_text_search", UNSET)

        search_image_url = d.pop("search_image_url", UNSET)

        search_video_url = d.pop("search_video_url", UNSET)

        search_3d_url = d.pop("search_3d_url", UNSET)

        search_audio_url = d.pop("search_audio_url", UNSET)

        document_browse_request = cls(
            top_k=top_k,
            cursor=cursor,
            include_attributes=include_attributes,
            filters=filters,
            search_text=search_text,
            use_hybrid_text_search=use_hybrid_text_search,
            search_image_url=search_image_url,
            search_video_url=search_video_url,
            search_3d_url=search_3d_url,
            search_audio_url=search_audio_url,
        )

        document_browse_request.additional_properties = d
        return document_browse_request

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

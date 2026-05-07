from typing import Any, TypeVar, Union, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="PublicRegistryEndpoint")


@_attrs_define
class PublicRegistryEndpoint:
    """
    Attributes:
        endpoint (str):
        title (str):
        thumbnail_url (str):
        model_url (str):
        model_playground_url (str):
        category (str):
        license_type (Union[Unset, str]):  Default: ''.
        minimum_credits_required (Union[Unset, int]):  Default: 0.
        tags (Union[Unset, list[str]]):
        group_key (Union[Unset, str]):
        group_label (Union[Unset, str]):
        description (Union[Unset, str]):  Default: ''.
        short_description (Union[Unset, str]):  Default: ''.
        duration_estimate (Union[Unset, int]):
    """

    endpoint: str
    title: str
    thumbnail_url: str
    model_url: str
    model_playground_url: str
    category: str
    license_type: Union[Unset, str] = ""
    minimum_credits_required: Union[Unset, int] = 0
    tags: Union[Unset, list[str]] = UNSET
    group_key: Union[Unset, str] = UNSET
    group_label: Union[Unset, str] = UNSET
    description: Union[Unset, str] = ""
    short_description: Union[Unset, str] = ""
    duration_estimate: Union[Unset, int] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        endpoint = self.endpoint

        title = self.title

        thumbnail_url = self.thumbnail_url

        model_url = self.model_url

        model_playground_url = self.model_playground_url

        category = self.category

        license_type = self.license_type

        minimum_credits_required = self.minimum_credits_required

        tags: Union[Unset, list[str]] = UNSET
        if not isinstance(self.tags, Unset):
            tags = self.tags

        group_key = self.group_key

        group_label = self.group_label

        description = self.description

        short_description = self.short_description

        duration_estimate = self.duration_estimate

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "endpoint": endpoint,
                "title": title,
                "thumbnail_url": thumbnail_url,
                "model_url": model_url,
                "model_playground_url": model_playground_url,
                "category": category,
            }
        )
        if license_type is not UNSET:
            field_dict["license_type"] = license_type
        if minimum_credits_required is not UNSET:
            field_dict["minimum_credits_required"] = minimum_credits_required
        if tags is not UNSET:
            field_dict["tags"] = tags
        if group_key is not UNSET:
            field_dict["group_key"] = group_key
        if group_label is not UNSET:
            field_dict["group_label"] = group_label
        if description is not UNSET:
            field_dict["description"] = description
        if short_description is not UNSET:
            field_dict["short_description"] = short_description
        if duration_estimate is not UNSET:
            field_dict["duration_estimate"] = duration_estimate

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        endpoint = d.pop("endpoint")

        title = d.pop("title")

        thumbnail_url = d.pop("thumbnail_url")

        model_url = d.pop("model_url")

        model_playground_url = d.pop("model_playground_url")

        category = d.pop("category")

        license_type = d.pop("license_type", UNSET)

        minimum_credits_required = d.pop("minimum_credits_required", UNSET)

        tags = cast(list[str], d.pop("tags", UNSET))

        group_key = d.pop("group_key", UNSET)

        group_label = d.pop("group_label", UNSET)

        description = d.pop("description", UNSET)

        short_description = d.pop("short_description", UNSET)

        duration_estimate = d.pop("duration_estimate", UNSET)

        public_registry_endpoint = cls(
            endpoint=endpoint,
            title=title,
            thumbnail_url=thumbnail_url,
            model_url=model_url,
            model_playground_url=model_playground_url,
            category=category,
            license_type=license_type,
            minimum_credits_required=minimum_credits_required,
            tags=tags,
            group_key=group_key,
            group_label=group_label,
            description=description,
            short_description=short_description,
            duration_estimate=duration_estimate,
        )

        public_registry_endpoint.additional_properties = d
        return public_registry_endpoint

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

import datetime
from typing import TYPE_CHECKING, Any, TypeVar, Union, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from dateutil.parser import isoparse

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.registry_endpoint_examples import RegistryEndpointExamples


T = TypeVar("T", bound="RegistryEndpoint")


@_attrs_define
class RegistryEndpoint:
    """
    Attributes:
        endpoint (str):
        title (str):
        created_at (datetime.datetime):
        created_by_user_id (str):
        thumbnail_url (str):
        model_url (str):
        category (str):
        github_url (Union[Unset, str]):  Default: ''.
        license_type (Union[Unset, str]):  Default: ''.
        minimum_credits_required (Union[Unset, int]):  Default: 0.
        tags (Union[Unset, list[str]]):
        group_key (Union[Unset, str]):
        group_label (Union[Unset, str]):
        pricing_text (Union[Unset, str]):
        billing_message (Union[Unset, str]):
        training_endpoints (Union[Unset, list[str]]):
        inference_endpoints (Union[Unset, list[str]]):
        kind (Union[Unset, str]):
        machine_type (Union[Unset, str]):
        examples (Union[Unset, RegistryEndpointExamples]):
        description (Union[Unset, str]):  Default: ''.
        short_description (Union[Unset, str]):  Default: ''.
        thumbnail_animated_url (Union[Unset, str]):
        stream_url (Union[Unset, str]):
        duration_estimate (Union[Unset, int]):
        ranking (Union[Unset, int]):  Default: 2.
        minumum_units (Union[Unset, int]):
        highlighted (Union[Unset, bool]):  Default: False.
        auth_skippable (Union[Unset, bool]):  Default: False.
        unlisted (Union[Unset, bool]):  Default: False.
        deprecated (Union[Unset, bool]):  Default: False.
        result_comparison (Union[Unset, bool]):  Default: False.
        hide_pricing (Union[Unset, bool]):  Default: False.
        private (Union[Unset, bool]):  Default: False.
        removed (Union[Unset, bool]):  Default: False.
        admin_only (Union[Unset, bool]):  Default: False.
    """

    endpoint: str
    title: str
    created_at: datetime.datetime
    created_by_user_id: str
    thumbnail_url: str
    model_url: str
    category: str
    github_url: Union[Unset, str] = ""
    license_type: Union[Unset, str] = ""
    minimum_credits_required: Union[Unset, int] = 0
    tags: Union[Unset, list[str]] = UNSET
    group_key: Union[Unset, str] = UNSET
    group_label: Union[Unset, str] = UNSET
    pricing_text: Union[Unset, str] = UNSET
    billing_message: Union[Unset, str] = UNSET
    training_endpoints: Union[Unset, list[str]] = UNSET
    inference_endpoints: Union[Unset, list[str]] = UNSET
    kind: Union[Unset, str] = UNSET
    machine_type: Union[Unset, str] = UNSET
    examples: Union[Unset, "RegistryEndpointExamples"] = UNSET
    description: Union[Unset, str] = ""
    short_description: Union[Unset, str] = ""
    thumbnail_animated_url: Union[Unset, str] = UNSET
    stream_url: Union[Unset, str] = UNSET
    duration_estimate: Union[Unset, int] = UNSET
    ranking: Union[Unset, int] = 2
    minumum_units: Union[Unset, int] = UNSET
    highlighted: Union[Unset, bool] = False
    auth_skippable: Union[Unset, bool] = False
    unlisted: Union[Unset, bool] = False
    deprecated: Union[Unset, bool] = False
    result_comparison: Union[Unset, bool] = False
    hide_pricing: Union[Unset, bool] = False
    private: Union[Unset, bool] = False
    removed: Union[Unset, bool] = False
    admin_only: Union[Unset, bool] = False
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        endpoint = self.endpoint

        title = self.title

        created_at = self.created_at.isoformat()

        created_by_user_id = self.created_by_user_id

        thumbnail_url = self.thumbnail_url

        model_url = self.model_url

        category = self.category

        github_url = self.github_url

        license_type = self.license_type

        minimum_credits_required = self.minimum_credits_required

        tags: Union[Unset, list[str]] = UNSET
        if not isinstance(self.tags, Unset):
            tags = self.tags

        group_key = self.group_key

        group_label = self.group_label

        pricing_text = self.pricing_text

        billing_message = self.billing_message

        training_endpoints: Union[Unset, list[str]] = UNSET
        if not isinstance(self.training_endpoints, Unset):
            training_endpoints = self.training_endpoints

        inference_endpoints: Union[Unset, list[str]] = UNSET
        if not isinstance(self.inference_endpoints, Unset):
            inference_endpoints = self.inference_endpoints

        kind = self.kind

        machine_type = self.machine_type

        examples: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.examples, Unset):
            examples = self.examples.to_dict()

        description = self.description

        short_description = self.short_description

        thumbnail_animated_url = self.thumbnail_animated_url

        stream_url = self.stream_url

        duration_estimate = self.duration_estimate

        ranking = self.ranking

        minumum_units = self.minumum_units

        highlighted = self.highlighted

        auth_skippable = self.auth_skippable

        unlisted = self.unlisted

        deprecated = self.deprecated

        result_comparison = self.result_comparison

        hide_pricing = self.hide_pricing

        private = self.private

        removed = self.removed

        admin_only = self.admin_only

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "endpoint": endpoint,
                "title": title,
                "created_at": created_at,
                "created_by_user_id": created_by_user_id,
                "thumbnail_url": thumbnail_url,
                "model_url": model_url,
                "category": category,
            }
        )
        if github_url is not UNSET:
            field_dict["github_url"] = github_url
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
        if pricing_text is not UNSET:
            field_dict["pricing_text"] = pricing_text
        if billing_message is not UNSET:
            field_dict["billing_message"] = billing_message
        if training_endpoints is not UNSET:
            field_dict["training_endpoints"] = training_endpoints
        if inference_endpoints is not UNSET:
            field_dict["inference_endpoints"] = inference_endpoints
        if kind is not UNSET:
            field_dict["kind"] = kind
        if machine_type is not UNSET:
            field_dict["machine_type"] = machine_type
        if examples is not UNSET:
            field_dict["examples"] = examples
        if description is not UNSET:
            field_dict["description"] = description
        if short_description is not UNSET:
            field_dict["short_description"] = short_description
        if thumbnail_animated_url is not UNSET:
            field_dict["thumbnail_animated_url"] = thumbnail_animated_url
        if stream_url is not UNSET:
            field_dict["stream_url"] = stream_url
        if duration_estimate is not UNSET:
            field_dict["duration_estimate"] = duration_estimate
        if ranking is not UNSET:
            field_dict["ranking"] = ranking
        if minumum_units is not UNSET:
            field_dict["minumum_units"] = minumum_units
        if highlighted is not UNSET:
            field_dict["highlighted"] = highlighted
        if auth_skippable is not UNSET:
            field_dict["auth_skippable"] = auth_skippable
        if unlisted is not UNSET:
            field_dict["unlisted"] = unlisted
        if deprecated is not UNSET:
            field_dict["deprecated"] = deprecated
        if result_comparison is not UNSET:
            field_dict["result_comparison"] = result_comparison
        if hide_pricing is not UNSET:
            field_dict["hide_pricing"] = hide_pricing
        if private is not UNSET:
            field_dict["private"] = private
        if removed is not UNSET:
            field_dict["removed"] = removed
        if admin_only is not UNSET:
            field_dict["admin_only"] = admin_only

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        from ..models.registry_endpoint_examples import RegistryEndpointExamples

        d = src_dict.copy()
        endpoint = d.pop("endpoint")

        title = d.pop("title")

        created_at = isoparse(d.pop("created_at"))

        created_by_user_id = d.pop("created_by_user_id")

        thumbnail_url = d.pop("thumbnail_url")

        model_url = d.pop("model_url")

        category = d.pop("category")

        github_url = d.pop("github_url", UNSET)

        license_type = d.pop("license_type", UNSET)

        minimum_credits_required = d.pop("minimum_credits_required", UNSET)

        tags = cast(list[str], d.pop("tags", UNSET))

        group_key = d.pop("group_key", UNSET)

        group_label = d.pop("group_label", UNSET)

        pricing_text = d.pop("pricing_text", UNSET)

        billing_message = d.pop("billing_message", UNSET)

        training_endpoints = cast(list[str], d.pop("training_endpoints", UNSET))

        inference_endpoints = cast(list[str], d.pop("inference_endpoints", UNSET))

        kind = d.pop("kind", UNSET)

        machine_type = d.pop("machine_type", UNSET)

        _examples = d.pop("examples", UNSET)
        examples: Union[Unset, RegistryEndpointExamples]
        if isinstance(_examples, Unset):
            examples = UNSET
        else:
            examples = RegistryEndpointExamples.from_dict(_examples)

        description = d.pop("description", UNSET)

        short_description = d.pop("short_description", UNSET)

        thumbnail_animated_url = d.pop("thumbnail_animated_url", UNSET)

        stream_url = d.pop("stream_url", UNSET)

        duration_estimate = d.pop("duration_estimate", UNSET)

        ranking = d.pop("ranking", UNSET)

        minumum_units = d.pop("minumum_units", UNSET)

        highlighted = d.pop("highlighted", UNSET)

        auth_skippable = d.pop("auth_skippable", UNSET)

        unlisted = d.pop("unlisted", UNSET)

        deprecated = d.pop("deprecated", UNSET)

        result_comparison = d.pop("result_comparison", UNSET)

        hide_pricing = d.pop("hide_pricing", UNSET)

        private = d.pop("private", UNSET)

        removed = d.pop("removed", UNSET)

        admin_only = d.pop("admin_only", UNSET)

        registry_endpoint = cls(
            endpoint=endpoint,
            title=title,
            created_at=created_at,
            created_by_user_id=created_by_user_id,
            thumbnail_url=thumbnail_url,
            model_url=model_url,
            category=category,
            github_url=github_url,
            license_type=license_type,
            minimum_credits_required=minimum_credits_required,
            tags=tags,
            group_key=group_key,
            group_label=group_label,
            pricing_text=pricing_text,
            billing_message=billing_message,
            training_endpoints=training_endpoints,
            inference_endpoints=inference_endpoints,
            kind=kind,
            machine_type=machine_type,
            examples=examples,
            description=description,
            short_description=short_description,
            thumbnail_animated_url=thumbnail_animated_url,
            stream_url=stream_url,
            duration_estimate=duration_estimate,
            ranking=ranking,
            minumum_units=minumum_units,
            highlighted=highlighted,
            auth_skippable=auth_skippable,
            unlisted=unlisted,
            deprecated=deprecated,
            result_comparison=result_comparison,
            hide_pricing=hide_pricing,
            private=private,
            removed=removed,
            admin_only=admin_only,
        )

        registry_endpoint.additional_properties = d
        return registry_endpoint

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

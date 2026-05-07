from typing import Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="PlatformAppQueueSizePayload")


@_attrs_define
class PlatformAppQueueSizePayload:
    """
    Attributes:
        application (Union[Unset, str]):
        full_application (Union[Unset, str]):
        environment (Union[Unset, str]):
        queue_size (Union[Unset, int]):
        queue_size_threshold (Union[Unset, int]):
        queue_size_threshold_duration (Union[Unset, str]):
    """

    application: Union[Unset, str] = UNSET
    full_application: Union[Unset, str] = UNSET
    environment: Union[Unset, str] = UNSET
    queue_size: Union[Unset, int] = UNSET
    queue_size_threshold: Union[Unset, int] = UNSET
    queue_size_threshold_duration: Union[Unset, str] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        application = self.application

        full_application = self.full_application

        environment = self.environment

        queue_size = self.queue_size

        queue_size_threshold = self.queue_size_threshold

        queue_size_threshold_duration = self.queue_size_threshold_duration

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if application is not UNSET:
            field_dict["application"] = application
        if full_application is not UNSET:
            field_dict["full_application"] = full_application
        if environment is not UNSET:
            field_dict["environment"] = environment
        if queue_size is not UNSET:
            field_dict["queue_size"] = queue_size
        if queue_size_threshold is not UNSET:
            field_dict["queue_size_threshold"] = queue_size_threshold
        if queue_size_threshold_duration is not UNSET:
            field_dict["queue_size_threshold_duration"] = queue_size_threshold_duration

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        application = d.pop("application", UNSET)

        full_application = d.pop("full_application", UNSET)

        environment = d.pop("environment", UNSET)

        queue_size = d.pop("queue_size", UNSET)

        queue_size_threshold = d.pop("queue_size_threshold", UNSET)

        queue_size_threshold_duration = d.pop("queue_size_threshold_duration", UNSET)

        platform_app_queue_size_payload = cls(
            application=application,
            full_application=full_application,
            environment=environment,
            queue_size=queue_size,
            queue_size_threshold=queue_size_threshold,
            queue_size_threshold_duration=queue_size_threshold_duration,
        )

        platform_app_queue_size_payload.additional_properties = d
        return platform_app_queue_size_payload

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

from typing import Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="PlatformAppHttp5XxErrorsPayload")


@_attrs_define
class PlatformAppHttp5XxErrorsPayload:
    """
    Attributes:
        application (Union[Unset, str]):
        full_application (Union[Unset, str]):
        environment (Union[Unset, str]):
        http_5xx_count (Union[Unset, float]):
        http_5xx_threshold (Union[Unset, int]):
        http_5xx_threshold_duration (Union[Unset, str]):
    """

    application: Union[Unset, str] = UNSET
    full_application: Union[Unset, str] = UNSET
    environment: Union[Unset, str] = UNSET
    http_5xx_count: Union[Unset, float] = UNSET
    http_5xx_threshold: Union[Unset, int] = UNSET
    http_5xx_threshold_duration: Union[Unset, str] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        application = self.application

        full_application = self.full_application

        environment = self.environment

        http_5xx_count = self.http_5xx_count

        http_5xx_threshold = self.http_5xx_threshold

        http_5xx_threshold_duration = self.http_5xx_threshold_duration

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if application is not UNSET:
            field_dict["application"] = application
        if full_application is not UNSET:
            field_dict["full_application"] = full_application
        if environment is not UNSET:
            field_dict["environment"] = environment
        if http_5xx_count is not UNSET:
            field_dict["http_5xx_count"] = http_5xx_count
        if http_5xx_threshold is not UNSET:
            field_dict["http_5xx_threshold"] = http_5xx_threshold
        if http_5xx_threshold_duration is not UNSET:
            field_dict["http_5xx_threshold_duration"] = http_5xx_threshold_duration

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        application = d.pop("application", UNSET)

        full_application = d.pop("full_application", UNSET)

        environment = d.pop("environment", UNSET)

        http_5xx_count = d.pop("http_5xx_count", UNSET)

        http_5xx_threshold = d.pop("http_5xx_threshold", UNSET)

        http_5xx_threshold_duration = d.pop("http_5xx_threshold_duration", UNSET)

        platform_app_http_5_xx_errors_payload = cls(
            application=application,
            full_application=full_application,
            environment=environment,
            http_5xx_count=http_5xx_count,
            http_5xx_threshold=http_5xx_threshold,
            http_5xx_threshold_duration=http_5xx_threshold_duration,
        )

        platform_app_http_5_xx_errors_payload.additional_properties = d
        return platform_app_http_5_xx_errors_payload

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

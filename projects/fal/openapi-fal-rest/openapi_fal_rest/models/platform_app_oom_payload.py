from typing import Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="PlatformAppOomPayload")


@_attrs_define
class PlatformAppOomPayload:
    """
    Attributes:
        application (Union[Unset, str]):
        full_application (Union[Unset, str]):
        environment (Union[Unset, str]):
        job_id (Union[Unset, str]):
        oom_kills_count (Union[Unset, int]):
        oom_kills_period (Union[Unset, str]):
    """

    application: Union[Unset, str] = UNSET
    full_application: Union[Unset, str] = UNSET
    environment: Union[Unset, str] = UNSET
    job_id: Union[Unset, str] = UNSET
    oom_kills_count: Union[Unset, int] = UNSET
    oom_kills_period: Union[Unset, str] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        application = self.application

        full_application = self.full_application

        environment = self.environment

        job_id = self.job_id

        oom_kills_count = self.oom_kills_count

        oom_kills_period = self.oom_kills_period

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if application is not UNSET:
            field_dict["application"] = application
        if full_application is not UNSET:
            field_dict["full_application"] = full_application
        if environment is not UNSET:
            field_dict["environment"] = environment
        if job_id is not UNSET:
            field_dict["job_id"] = job_id
        if oom_kills_count is not UNSET:
            field_dict["oom_kills_count"] = oom_kills_count
        if oom_kills_period is not UNSET:
            field_dict["oom_kills_period"] = oom_kills_period

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        application = d.pop("application", UNSET)

        full_application = d.pop("full_application", UNSET)

        environment = d.pop("environment", UNSET)

        job_id = d.pop("job_id", UNSET)

        oom_kills_count = d.pop("oom_kills_count", UNSET)

        oom_kills_period = d.pop("oom_kills_period", UNSET)

        platform_app_oom_payload = cls(
            application=application,
            full_application=full_application,
            environment=environment,
            job_id=job_id,
            oom_kills_count=oom_kills_count,
            oom_kills_period=oom_kills_period,
        )

        platform_app_oom_payload.additional_properties = d
        return platform_app_oom_payload

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

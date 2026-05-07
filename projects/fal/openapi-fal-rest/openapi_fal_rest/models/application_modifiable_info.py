from typing import TYPE_CHECKING, Any, TypeVar, Union, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.application_modifiable_info_max_gpus_per_user_by_type import (
        ApplicationModifiableInfoMaxGpusPerUserByType,
    )


T = TypeVar("T", bound="ApplicationModifiableInfo")


@_attrs_define
class ApplicationModifiableInfo:
    """
    Attributes:
        keep_alive (int):
        max_concurrency (int):
        max_multiplexing (int):
        min_concurrency (int):
        max_gpus_per_user (int):
        max_gpus_per_user_by_type (ApplicationModifiableInfoMaxGpusPerUserByType):
        concurrency_buffer (int):
        concurrency_buffer_perc (int):
        scaling_delay_seconds (int):
        valid_regions (list[str]):
        alias_set (bool):
        request_timeout (int):
        startup_timeout (int):
        machine_type (str):
        is_locked (Union[Unset, bool]):  Default: False.
    """

    keep_alive: int
    max_concurrency: int
    max_multiplexing: int
    min_concurrency: int
    max_gpus_per_user: int
    max_gpus_per_user_by_type: "ApplicationModifiableInfoMaxGpusPerUserByType"
    concurrency_buffer: int
    concurrency_buffer_perc: int
    scaling_delay_seconds: int
    valid_regions: list[str]
    alias_set: bool
    request_timeout: int
    startup_timeout: int
    machine_type: str
    is_locked: Union[Unset, bool] = False
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        keep_alive = self.keep_alive

        max_concurrency = self.max_concurrency

        max_multiplexing = self.max_multiplexing

        min_concurrency = self.min_concurrency

        max_gpus_per_user = self.max_gpus_per_user

        max_gpus_per_user_by_type = self.max_gpus_per_user_by_type.to_dict()

        concurrency_buffer = self.concurrency_buffer

        concurrency_buffer_perc = self.concurrency_buffer_perc

        scaling_delay_seconds = self.scaling_delay_seconds

        valid_regions = self.valid_regions

        alias_set = self.alias_set

        request_timeout = self.request_timeout

        startup_timeout = self.startup_timeout

        machine_type = self.machine_type

        is_locked = self.is_locked

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "keep_alive": keep_alive,
                "max_concurrency": max_concurrency,
                "max_multiplexing": max_multiplexing,
                "min_concurrency": min_concurrency,
                "max_gpus_per_user": max_gpus_per_user,
                "max_gpus_per_user_by_type": max_gpus_per_user_by_type,
                "concurrency_buffer": concurrency_buffer,
                "concurrency_buffer_perc": concurrency_buffer_perc,
                "scaling_delay_seconds": scaling_delay_seconds,
                "valid_regions": valid_regions,
                "alias_set": alias_set,
                "request_timeout": request_timeout,
                "startup_timeout": startup_timeout,
                "machine_type": machine_type,
            }
        )
        if is_locked is not UNSET:
            field_dict["is_locked"] = is_locked

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        from ..models.application_modifiable_info_max_gpus_per_user_by_type import (
            ApplicationModifiableInfoMaxGpusPerUserByType,
        )

        d = src_dict.copy()
        keep_alive = d.pop("keep_alive")

        max_concurrency = d.pop("max_concurrency")

        max_multiplexing = d.pop("max_multiplexing")

        min_concurrency = d.pop("min_concurrency")

        max_gpus_per_user = d.pop("max_gpus_per_user")

        max_gpus_per_user_by_type = ApplicationModifiableInfoMaxGpusPerUserByType.from_dict(
            d.pop("max_gpus_per_user_by_type")
        )

        concurrency_buffer = d.pop("concurrency_buffer")

        concurrency_buffer_perc = d.pop("concurrency_buffer_perc")

        scaling_delay_seconds = d.pop("scaling_delay_seconds")

        valid_regions = cast(list[str], d.pop("valid_regions"))

        alias_set = d.pop("alias_set")

        request_timeout = d.pop("request_timeout")

        startup_timeout = d.pop("startup_timeout")

        machine_type = d.pop("machine_type")

        is_locked = d.pop("is_locked", UNSET)

        application_modifiable_info = cls(
            keep_alive=keep_alive,
            max_concurrency=max_concurrency,
            max_multiplexing=max_multiplexing,
            min_concurrency=min_concurrency,
            max_gpus_per_user=max_gpus_per_user,
            max_gpus_per_user_by_type=max_gpus_per_user_by_type,
            concurrency_buffer=concurrency_buffer,
            concurrency_buffer_perc=concurrency_buffer_perc,
            scaling_delay_seconds=scaling_delay_seconds,
            valid_regions=valid_regions,
            alias_set=alias_set,
            request_timeout=request_timeout,
            startup_timeout=startup_timeout,
            machine_type=machine_type,
            is_locked=is_locked,
        )

        application_modifiable_info.additional_properties = d
        return application_modifiable_info

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

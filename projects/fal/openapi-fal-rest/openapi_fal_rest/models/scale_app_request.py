from typing import Any, TypeVar, Union, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="ScaleAppRequest")


@_attrs_define
class ScaleAppRequest:
    """
    Attributes:
        keep_alive (Union[Unset, int]):
        max_multiplexing (Union[Unset, int]):
        max_concurrency (Union[Unset, int]):
        min_concurrency (Union[Unset, int]):
        concurrency_buffer (Union[Unset, int]):
        concurrency_buffer_perc (Union[Unset, int]):
        scaling_delay_seconds (Union[Unset, int]):
        machine_type (Union[Unset, str]):
        request_timeout (Union[Unset, int]):
        startup_timeout (Union[Unset, int]):
        valid_regions (Union[Unset, list[str]]):
    """

    keep_alive: Union[Unset, int] = UNSET
    max_multiplexing: Union[Unset, int] = UNSET
    max_concurrency: Union[Unset, int] = UNSET
    min_concurrency: Union[Unset, int] = UNSET
    concurrency_buffer: Union[Unset, int] = UNSET
    concurrency_buffer_perc: Union[Unset, int] = UNSET
    scaling_delay_seconds: Union[Unset, int] = UNSET
    machine_type: Union[Unset, str] = UNSET
    request_timeout: Union[Unset, int] = UNSET
    startup_timeout: Union[Unset, int] = UNSET
    valid_regions: Union[Unset, list[str]] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        keep_alive = self.keep_alive

        max_multiplexing = self.max_multiplexing

        max_concurrency = self.max_concurrency

        min_concurrency = self.min_concurrency

        concurrency_buffer = self.concurrency_buffer

        concurrency_buffer_perc = self.concurrency_buffer_perc

        scaling_delay_seconds = self.scaling_delay_seconds

        machine_type = self.machine_type

        request_timeout = self.request_timeout

        startup_timeout = self.startup_timeout

        valid_regions: Union[Unset, list[str]] = UNSET
        if not isinstance(self.valid_regions, Unset):
            valid_regions = self.valid_regions

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if keep_alive is not UNSET:
            field_dict["keep_alive"] = keep_alive
        if max_multiplexing is not UNSET:
            field_dict["max_multiplexing"] = max_multiplexing
        if max_concurrency is not UNSET:
            field_dict["max_concurrency"] = max_concurrency
        if min_concurrency is not UNSET:
            field_dict["min_concurrency"] = min_concurrency
        if concurrency_buffer is not UNSET:
            field_dict["concurrency_buffer"] = concurrency_buffer
        if concurrency_buffer_perc is not UNSET:
            field_dict["concurrency_buffer_perc"] = concurrency_buffer_perc
        if scaling_delay_seconds is not UNSET:
            field_dict["scaling_delay_seconds"] = scaling_delay_seconds
        if machine_type is not UNSET:
            field_dict["machine_type"] = machine_type
        if request_timeout is not UNSET:
            field_dict["request_timeout"] = request_timeout
        if startup_timeout is not UNSET:
            field_dict["startup_timeout"] = startup_timeout
        if valid_regions is not UNSET:
            field_dict["valid_regions"] = valid_regions

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        keep_alive = d.pop("keep_alive", UNSET)

        max_multiplexing = d.pop("max_multiplexing", UNSET)

        max_concurrency = d.pop("max_concurrency", UNSET)

        min_concurrency = d.pop("min_concurrency", UNSET)

        concurrency_buffer = d.pop("concurrency_buffer", UNSET)

        concurrency_buffer_perc = d.pop("concurrency_buffer_perc", UNSET)

        scaling_delay_seconds = d.pop("scaling_delay_seconds", UNSET)

        machine_type = d.pop("machine_type", UNSET)

        request_timeout = d.pop("request_timeout", UNSET)

        startup_timeout = d.pop("startup_timeout", UNSET)

        valid_regions = cast(list[str], d.pop("valid_regions", UNSET))

        scale_app_request = cls(
            keep_alive=keep_alive,
            max_multiplexing=max_multiplexing,
            max_concurrency=max_concurrency,
            min_concurrency=min_concurrency,
            concurrency_buffer=concurrency_buffer,
            concurrency_buffer_perc=concurrency_buffer_perc,
            scaling_delay_seconds=scaling_delay_seconds,
            machine_type=machine_type,
            request_timeout=request_timeout,
            startup_timeout=startup_timeout,
            valid_regions=valid_regions,
        )

        scale_app_request.additional_properties = d
        return scale_app_request

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

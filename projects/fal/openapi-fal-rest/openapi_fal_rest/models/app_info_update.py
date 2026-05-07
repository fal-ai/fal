from typing import Any, TypeVar, Union, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.application_auth_mode import ApplicationAuthMode
from ..types import UNSET, Unset

T = TypeVar("T", bound="AppInfoUpdate")


@_attrs_define
class AppInfoUpdate:
    """
    Attributes:
        machine_type (str):
        keep_alive (int):
        min_concurrency (int):
        max_concurrency (int):
        concurrency_buffer (int):
        max_multiplexing (int):
        valid_regions (list[str]):
        request_timeout (Union[Unset, int]):
        startup_timeout (Union[Unset, int]):
        concurrency_buffer_perc (Union[Unset, int]):
        scaling_delay_seconds (Union[Unset, int]):
        app_name (Union[Unset, str]):
        auth_mode (Union[Unset, ApplicationAuthMode]):
        application_id (Union[Unset, str]):
        active_runners (Union[Unset, int]):
    """

    machine_type: str
    keep_alive: int
    min_concurrency: int
    max_concurrency: int
    concurrency_buffer: int
    max_multiplexing: int
    valid_regions: list[str]
    request_timeout: Union[Unset, int] = UNSET
    startup_timeout: Union[Unset, int] = UNSET
    concurrency_buffer_perc: Union[Unset, int] = UNSET
    scaling_delay_seconds: Union[Unset, int] = UNSET
    app_name: Union[Unset, str] = UNSET
    auth_mode: Union[Unset, ApplicationAuthMode] = UNSET
    application_id: Union[Unset, str] = UNSET
    active_runners: Union[Unset, int] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        machine_type = self.machine_type

        keep_alive = self.keep_alive

        min_concurrency = self.min_concurrency

        max_concurrency = self.max_concurrency

        concurrency_buffer = self.concurrency_buffer

        max_multiplexing = self.max_multiplexing

        valid_regions = self.valid_regions

        request_timeout = self.request_timeout

        startup_timeout = self.startup_timeout

        concurrency_buffer_perc = self.concurrency_buffer_perc

        scaling_delay_seconds = self.scaling_delay_seconds

        app_name = self.app_name

        auth_mode: Union[Unset, str] = UNSET
        if not isinstance(self.auth_mode, Unset):
            auth_mode = self.auth_mode.value

        application_id = self.application_id

        active_runners = self.active_runners

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "machine_type": machine_type,
                "keep_alive": keep_alive,
                "min_concurrency": min_concurrency,
                "max_concurrency": max_concurrency,
                "concurrency_buffer": concurrency_buffer,
                "max_multiplexing": max_multiplexing,
                "valid_regions": valid_regions,
            }
        )
        if request_timeout is not UNSET:
            field_dict["request_timeout"] = request_timeout
        if startup_timeout is not UNSET:
            field_dict["startup_timeout"] = startup_timeout
        if concurrency_buffer_perc is not UNSET:
            field_dict["concurrency_buffer_perc"] = concurrency_buffer_perc
        if scaling_delay_seconds is not UNSET:
            field_dict["scaling_delay_seconds"] = scaling_delay_seconds
        if app_name is not UNSET:
            field_dict["app_name"] = app_name
        if auth_mode is not UNSET:
            field_dict["auth_mode"] = auth_mode
        if application_id is not UNSET:
            field_dict["application_id"] = application_id
        if active_runners is not UNSET:
            field_dict["active_runners"] = active_runners

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        machine_type = d.pop("machine_type")

        keep_alive = d.pop("keep_alive")

        min_concurrency = d.pop("min_concurrency")

        max_concurrency = d.pop("max_concurrency")

        concurrency_buffer = d.pop("concurrency_buffer")

        max_multiplexing = d.pop("max_multiplexing")

        valid_regions = cast(list[str], d.pop("valid_regions"))

        request_timeout = d.pop("request_timeout", UNSET)

        startup_timeout = d.pop("startup_timeout", UNSET)

        concurrency_buffer_perc = d.pop("concurrency_buffer_perc", UNSET)

        scaling_delay_seconds = d.pop("scaling_delay_seconds", UNSET)

        app_name = d.pop("app_name", UNSET)

        _auth_mode = d.pop("auth_mode", UNSET)
        auth_mode: Union[Unset, ApplicationAuthMode]
        if isinstance(_auth_mode, Unset):
            auth_mode = UNSET
        else:
            auth_mode = ApplicationAuthMode(_auth_mode)

        application_id = d.pop("application_id", UNSET)

        active_runners = d.pop("active_runners", UNSET)

        app_info_update = cls(
            machine_type=machine_type,
            keep_alive=keep_alive,
            min_concurrency=min_concurrency,
            max_concurrency=max_concurrency,
            concurrency_buffer=concurrency_buffer,
            max_multiplexing=max_multiplexing,
            valid_regions=valid_regions,
            request_timeout=request_timeout,
            startup_timeout=startup_timeout,
            concurrency_buffer_perc=concurrency_buffer_perc,
            scaling_delay_seconds=scaling_delay_seconds,
            app_name=app_name,
            auth_mode=auth_mode,
            application_id=application_id,
            active_runners=active_runners,
        )

        app_info_update.additional_properties = d
        return app_info_update

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

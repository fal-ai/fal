from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.application_auth_mode import ApplicationAuthMode

if TYPE_CHECKING:
    from ..models.app_info_per_user_auth import AppInfoPerUserAuth


T = TypeVar("T", bound="AppInfo")


@_attrs_define
class AppInfo:
    """
    Attributes:
        app_name (str):
        auth_mode (ApplicationAuthMode):
        per_user_auth (AppInfoPerUserAuth):
        machine_type (str):
        container_image (str):
        application_id (str):
        keep_alive (int):
        min_concurrency (int):
        max_concurrency (int):
        concurrency_buffer (int):
        concurrency_buffer_perc (int):
        scaling_delay_seconds (int):
        max_multiplexing (int):
        active_runners (int):
        valid_regions (list[str]):
        request_timeout (int):
        startup_timeout (int):
        environment_name (str):
    """

    app_name: str
    auth_mode: ApplicationAuthMode
    per_user_auth: "AppInfoPerUserAuth"
    machine_type: str
    container_image: str
    application_id: str
    keep_alive: int
    min_concurrency: int
    max_concurrency: int
    concurrency_buffer: int
    concurrency_buffer_perc: int
    scaling_delay_seconds: int
    max_multiplexing: int
    active_runners: int
    valid_regions: list[str]
    request_timeout: int
    startup_timeout: int
    environment_name: str
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        app_name = self.app_name

        auth_mode = self.auth_mode.value

        per_user_auth = self.per_user_auth.to_dict()

        machine_type = self.machine_type

        container_image = self.container_image

        application_id = self.application_id

        keep_alive = self.keep_alive

        min_concurrency = self.min_concurrency

        max_concurrency = self.max_concurrency

        concurrency_buffer = self.concurrency_buffer

        concurrency_buffer_perc = self.concurrency_buffer_perc

        scaling_delay_seconds = self.scaling_delay_seconds

        max_multiplexing = self.max_multiplexing

        active_runners = self.active_runners

        valid_regions = self.valid_regions

        request_timeout = self.request_timeout

        startup_timeout = self.startup_timeout

        environment_name = self.environment_name

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "app_name": app_name,
                "auth_mode": auth_mode,
                "per_user_auth": per_user_auth,
                "machine_type": machine_type,
                "container_image": container_image,
                "application_id": application_id,
                "keep_alive": keep_alive,
                "min_concurrency": min_concurrency,
                "max_concurrency": max_concurrency,
                "concurrency_buffer": concurrency_buffer,
                "concurrency_buffer_perc": concurrency_buffer_perc,
                "scaling_delay_seconds": scaling_delay_seconds,
                "max_multiplexing": max_multiplexing,
                "active_runners": active_runners,
                "valid_regions": valid_regions,
                "request_timeout": request_timeout,
                "startup_timeout": startup_timeout,
                "environment_name": environment_name,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        from ..models.app_info_per_user_auth import AppInfoPerUserAuth

        d = src_dict.copy()
        app_name = d.pop("app_name")

        auth_mode = ApplicationAuthMode(d.pop("auth_mode"))

        per_user_auth = AppInfoPerUserAuth.from_dict(d.pop("per_user_auth"))

        machine_type = d.pop("machine_type")

        container_image = d.pop("container_image")

        application_id = d.pop("application_id")

        keep_alive = d.pop("keep_alive")

        min_concurrency = d.pop("min_concurrency")

        max_concurrency = d.pop("max_concurrency")

        concurrency_buffer = d.pop("concurrency_buffer")

        concurrency_buffer_perc = d.pop("concurrency_buffer_perc")

        scaling_delay_seconds = d.pop("scaling_delay_seconds")

        max_multiplexing = d.pop("max_multiplexing")

        active_runners = d.pop("active_runners")

        valid_regions = cast(list[str], d.pop("valid_regions"))

        request_timeout = d.pop("request_timeout")

        startup_timeout = d.pop("startup_timeout")

        environment_name = d.pop("environment_name")

        app_info = cls(
            app_name=app_name,
            auth_mode=auth_mode,
            per_user_auth=per_user_auth,
            machine_type=machine_type,
            container_image=container_image,
            application_id=application_id,
            keep_alive=keep_alive,
            min_concurrency=min_concurrency,
            max_concurrency=max_concurrency,
            concurrency_buffer=concurrency_buffer,
            concurrency_buffer_perc=concurrency_buffer_perc,
            scaling_delay_seconds=scaling_delay_seconds,
            max_multiplexing=max_multiplexing,
            active_runners=active_runners,
            valid_regions=valid_regions,
            request_timeout=request_timeout,
            startup_timeout=startup_timeout,
            environment_name=environment_name,
        )

        app_info.additional_properties = d
        return app_info

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

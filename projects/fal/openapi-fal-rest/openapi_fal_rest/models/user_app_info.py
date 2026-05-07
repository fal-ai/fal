import datetime
from typing import TYPE_CHECKING, Any, TypeVar, Union, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from dateutil.parser import isoparse

from ..models.application_auth_mode import ApplicationAuthMode
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.user_app_info_metadata import UserAppInfoMetadata
    from ..models.user_app_info_user_auth_mode import UserAppInfoUserAuthMode


T = TypeVar("T", bound="UserAppInfo")


@_attrs_define
class UserAppInfo:
    """
    Attributes:
        app_name (str):
        app_user_id (str):
        app_user_nickname (str):
        auth_mode (ApplicationAuthMode):
        container_image (str):
        machine_type (str):
        application_id (str):
        keep_alive (int):
        min_concurrency (int):
        max_concurrency (int):
        concurrency_buffer (int):
        concurrency_buffer_perc (int):
        scaling_delay_seconds (int):
        max_multiplexing (int):
        request_timeout (int):
        startup_timeout (int):
        valid_regions (list[str]):
        updated_at (datetime.datetime):
        environment_name (str):
        is_namespace_aliased (bool):
        metadata (Union[Unset, UserAppInfoMetadata]):
        user_auth_mode (Union[Unset, UserAppInfoUserAuthMode]):
        source_image (Union[Unset, str]):
    """

    app_name: str
    app_user_id: str
    app_user_nickname: str
    auth_mode: ApplicationAuthMode
    container_image: str
    machine_type: str
    application_id: str
    keep_alive: int
    min_concurrency: int
    max_concurrency: int
    concurrency_buffer: int
    concurrency_buffer_perc: int
    scaling_delay_seconds: int
    max_multiplexing: int
    request_timeout: int
    startup_timeout: int
    valid_regions: list[str]
    updated_at: datetime.datetime
    environment_name: str
    is_namespace_aliased: bool
    metadata: Union[Unset, "UserAppInfoMetadata"] = UNSET
    user_auth_mode: Union[Unset, "UserAppInfoUserAuthMode"] = UNSET
    source_image: Union[Unset, str] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        app_name = self.app_name

        app_user_id = self.app_user_id

        app_user_nickname = self.app_user_nickname

        auth_mode = self.auth_mode.value

        container_image = self.container_image

        machine_type = self.machine_type

        application_id = self.application_id

        keep_alive = self.keep_alive

        min_concurrency = self.min_concurrency

        max_concurrency = self.max_concurrency

        concurrency_buffer = self.concurrency_buffer

        concurrency_buffer_perc = self.concurrency_buffer_perc

        scaling_delay_seconds = self.scaling_delay_seconds

        max_multiplexing = self.max_multiplexing

        request_timeout = self.request_timeout

        startup_timeout = self.startup_timeout

        valid_regions = self.valid_regions

        updated_at = self.updated_at.isoformat()

        environment_name = self.environment_name

        is_namespace_aliased = self.is_namespace_aliased

        metadata: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.metadata, Unset):
            metadata = self.metadata.to_dict()

        user_auth_mode: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.user_auth_mode, Unset):
            user_auth_mode = self.user_auth_mode.to_dict()

        source_image = self.source_image

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "app_name": app_name,
                "app_user_id": app_user_id,
                "app_user_nickname": app_user_nickname,
                "auth_mode": auth_mode,
                "container_image": container_image,
                "machine_type": machine_type,
                "application_id": application_id,
                "keep_alive": keep_alive,
                "min_concurrency": min_concurrency,
                "max_concurrency": max_concurrency,
                "concurrency_buffer": concurrency_buffer,
                "concurrency_buffer_perc": concurrency_buffer_perc,
                "scaling_delay_seconds": scaling_delay_seconds,
                "max_multiplexing": max_multiplexing,
                "request_timeout": request_timeout,
                "startup_timeout": startup_timeout,
                "valid_regions": valid_regions,
                "updated_at": updated_at,
                "environment_name": environment_name,
                "is_namespace_aliased": is_namespace_aliased,
            }
        )
        if metadata is not UNSET:
            field_dict["metadata"] = metadata
        if user_auth_mode is not UNSET:
            field_dict["user_auth_mode"] = user_auth_mode
        if source_image is not UNSET:
            field_dict["source_image"] = source_image

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        from ..models.user_app_info_metadata import UserAppInfoMetadata
        from ..models.user_app_info_user_auth_mode import UserAppInfoUserAuthMode

        d = src_dict.copy()
        app_name = d.pop("app_name")

        app_user_id = d.pop("app_user_id")

        app_user_nickname = d.pop("app_user_nickname")

        auth_mode = ApplicationAuthMode(d.pop("auth_mode"))

        container_image = d.pop("container_image")

        machine_type = d.pop("machine_type")

        application_id = d.pop("application_id")

        keep_alive = d.pop("keep_alive")

        min_concurrency = d.pop("min_concurrency")

        max_concurrency = d.pop("max_concurrency")

        concurrency_buffer = d.pop("concurrency_buffer")

        concurrency_buffer_perc = d.pop("concurrency_buffer_perc")

        scaling_delay_seconds = d.pop("scaling_delay_seconds")

        max_multiplexing = d.pop("max_multiplexing")

        request_timeout = d.pop("request_timeout")

        startup_timeout = d.pop("startup_timeout")

        valid_regions = cast(list[str], d.pop("valid_regions"))

        updated_at = isoparse(d.pop("updated_at"))

        environment_name = d.pop("environment_name")

        is_namespace_aliased = d.pop("is_namespace_aliased")

        _metadata = d.pop("metadata", UNSET)
        metadata: Union[Unset, UserAppInfoMetadata]
        if isinstance(_metadata, Unset):
            metadata = UNSET
        else:
            metadata = UserAppInfoMetadata.from_dict(_metadata)

        _user_auth_mode = d.pop("user_auth_mode", UNSET)
        user_auth_mode: Union[Unset, UserAppInfoUserAuthMode]
        if isinstance(_user_auth_mode, Unset):
            user_auth_mode = UNSET
        else:
            user_auth_mode = UserAppInfoUserAuthMode.from_dict(_user_auth_mode)

        source_image = d.pop("source_image", UNSET)

        user_app_info = cls(
            app_name=app_name,
            app_user_id=app_user_id,
            app_user_nickname=app_user_nickname,
            auth_mode=auth_mode,
            container_image=container_image,
            machine_type=machine_type,
            application_id=application_id,
            keep_alive=keep_alive,
            min_concurrency=min_concurrency,
            max_concurrency=max_concurrency,
            concurrency_buffer=concurrency_buffer,
            concurrency_buffer_perc=concurrency_buffer_perc,
            scaling_delay_seconds=scaling_delay_seconds,
            max_multiplexing=max_multiplexing,
            request_timeout=request_timeout,
            startup_timeout=startup_timeout,
            valid_regions=valid_regions,
            updated_at=updated_at,
            environment_name=environment_name,
            is_namespace_aliased=is_namespace_aliased,
            metadata=metadata,
            user_auth_mode=user_auth_mode,
            source_image=source_image,
        )

        user_app_info.additional_properties = d
        return user_app_info

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

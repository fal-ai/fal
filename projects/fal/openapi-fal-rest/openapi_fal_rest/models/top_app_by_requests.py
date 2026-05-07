from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

T = TypeVar("T", bound="TopAppByRequests")


@_attrs_define
class TopAppByRequests:
    """
    Attributes:
        app_name (str):
        app_user_id (str):
        app_user_nickname (str):
        request_count (int):
    """

    app_name: str
    app_user_id: str
    app_user_nickname: str
    request_count: int
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        app_name = self.app_name

        app_user_id = self.app_user_id

        app_user_nickname = self.app_user_nickname

        request_count = self.request_count

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "app_name": app_name,
                "app_user_id": app_user_id,
                "app_user_nickname": app_user_nickname,
                "request_count": request_count,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        app_name = d.pop("app_name")

        app_user_id = d.pop("app_user_id")

        app_user_nickname = d.pop("app_user_nickname")

        request_count = d.pop("request_count")

        top_app_by_requests = cls(
            app_name=app_name,
            app_user_id=app_user_id,
            app_user_nickname=app_user_nickname,
            request_count=request_count,
        )

        top_app_by_requests.additional_properties = d
        return top_app_by_requests

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

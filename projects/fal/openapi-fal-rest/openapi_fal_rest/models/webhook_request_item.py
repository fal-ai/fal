import datetime
from typing import Any, TypeVar, Union
from uuid import UUID

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from dateutil.parser import isoparse

from ..types import UNSET, Unset

T = TypeVar("T", bound="WebhookRequestItem")


@_attrs_define
class WebhookRequestItem:
    """
    Attributes:
        request_id (UUID):
        webhook_url (str):
        created_at (datetime.datetime):
        caller_user_id (Union[Unset, str]):
        status_code (Union[Unset, int]):
        endpoint (Union[Unset, str]):
    """

    request_id: UUID
    webhook_url: str
    created_at: datetime.datetime
    caller_user_id: Union[Unset, str] = UNSET
    status_code: Union[Unset, int] = UNSET
    endpoint: Union[Unset, str] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        request_id = str(self.request_id)

        webhook_url = self.webhook_url

        created_at = self.created_at.isoformat()

        caller_user_id = self.caller_user_id

        status_code = self.status_code

        endpoint = self.endpoint

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "request_id": request_id,
                "webhook_url": webhook_url,
                "created_at": created_at,
            }
        )
        if caller_user_id is not UNSET:
            field_dict["caller_user_id"] = caller_user_id
        if status_code is not UNSET:
            field_dict["status_code"] = status_code
        if endpoint is not UNSET:
            field_dict["endpoint"] = endpoint

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        request_id = UUID(d.pop("request_id"))

        webhook_url = d.pop("webhook_url")

        created_at = isoparse(d.pop("created_at"))

        caller_user_id = d.pop("caller_user_id", UNSET)

        status_code = d.pop("status_code", UNSET)

        endpoint = d.pop("endpoint", UNSET)

        webhook_request_item = cls(
            request_id=request_id,
            webhook_url=webhook_url,
            created_at=created_at,
            caller_user_id=caller_user_id,
            status_code=status_code,
            endpoint=endpoint,
        )

        webhook_request_item.additional_properties = d
        return webhook_request_item

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

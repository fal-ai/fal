import datetime
from typing import Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from dateutil.parser import isoparse

from ..models.endpoint_access_control_context import EndpointAccessControlContext
from ..models.resolved_endpoint_access_control_status import ResolvedEndpointAccessControlStatus
from ..types import UNSET, Unset

T = TypeVar("T", bound="EndpointAccessControl")


@_attrs_define
class EndpointAccessControl:
    """
    Attributes:
        endpoint (str):
        context (EndpointAccessControlContext): API: Direct API requests using API keys
            UI: Web UI requests (Playground, Sandbox, Workflows) via proxy (X-Fal-Playground header)
        status (ResolvedEndpointAccessControlStatus): Resolved status after inheritance - excludes INHERIT.
        created_at (datetime.datetime):
        updated_at (datetime.datetime):
        reason (Union[Unset, str]):
        last_edited_by_auth_id (Union[Unset, str]):
        inherited (Union[Unset, bool]):  Default: False.
    """

    endpoint: str
    context: EndpointAccessControlContext
    status: ResolvedEndpointAccessControlStatus
    created_at: datetime.datetime
    updated_at: datetime.datetime
    reason: Union[Unset, str] = UNSET
    last_edited_by_auth_id: Union[Unset, str] = UNSET
    inherited: Union[Unset, bool] = False
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        endpoint = self.endpoint

        context = self.context.value

        status = self.status.value

        created_at = self.created_at.isoformat()

        updated_at = self.updated_at.isoformat()

        reason = self.reason

        last_edited_by_auth_id = self.last_edited_by_auth_id

        inherited = self.inherited

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "endpoint": endpoint,
                "context": context,
                "status": status,
                "created_at": created_at,
                "updated_at": updated_at,
            }
        )
        if reason is not UNSET:
            field_dict["reason"] = reason
        if last_edited_by_auth_id is not UNSET:
            field_dict["last_edited_by_auth_id"] = last_edited_by_auth_id
        if inherited is not UNSET:
            field_dict["inherited"] = inherited

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        endpoint = d.pop("endpoint")

        context = EndpointAccessControlContext(d.pop("context"))

        status = ResolvedEndpointAccessControlStatus(d.pop("status"))

        created_at = isoparse(d.pop("created_at"))

        updated_at = isoparse(d.pop("updated_at"))

        reason = d.pop("reason", UNSET)

        last_edited_by_auth_id = d.pop("last_edited_by_auth_id", UNSET)

        inherited = d.pop("inherited", UNSET)

        endpoint_access_control = cls(
            endpoint=endpoint,
            context=context,
            status=status,
            created_at=created_at,
            updated_at=updated_at,
            reason=reason,
            last_edited_by_auth_id=last_edited_by_auth_id,
            inherited=inherited,
        )

        endpoint_access_control.additional_properties = d
        return endpoint_access_control

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

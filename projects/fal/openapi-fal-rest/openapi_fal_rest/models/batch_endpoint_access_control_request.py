from typing import Any, TypeVar, Union, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.endpoint_access_control_context import EndpointAccessControlContext
from ..models.endpoint_access_control_status import EndpointAccessControlStatus
from ..types import UNSET, Unset

T = TypeVar("T", bound="BatchEndpointAccessControlRequest")


@_attrs_define
class BatchEndpointAccessControlRequest:
    """
    Attributes:
        endpoints (list[str]):
        context (EndpointAccessControlContext): API: Direct API requests using API keys
            UI: Web UI requests (Playground, Sandbox, Workflows) via proxy (X-Fal-Playground header)
        status (EndpointAccessControlStatus):
        reason (Union[Unset, str]):
    """

    endpoints: list[str]
    context: EndpointAccessControlContext
    status: EndpointAccessControlStatus
    reason: Union[Unset, str] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        endpoints = self.endpoints

        context = self.context.value

        status = self.status.value

        reason = self.reason

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "endpoints": endpoints,
                "context": context,
                "status": status,
            }
        )
        if reason is not UNSET:
            field_dict["reason"] = reason

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        endpoints = cast(list[str], d.pop("endpoints"))

        context = EndpointAccessControlContext(d.pop("context"))

        status = EndpointAccessControlStatus(d.pop("status"))

        reason = d.pop("reason", UNSET)

        batch_endpoint_access_control_request = cls(
            endpoints=endpoints,
            context=context,
            status=status,
            reason=reason,
        )

        batch_endpoint_access_control_request.additional_properties = d
        return batch_endpoint_access_control_request

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

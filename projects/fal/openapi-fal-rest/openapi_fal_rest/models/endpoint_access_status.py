from typing import Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.enterprise_status import EnterpriseStatus
from ..types import UNSET, Unset

T = TypeVar("T", bound="EndpointAccessStatus")


@_attrs_define
class EndpointAccessStatus:
    """
    Attributes:
        endpoint (str):
        is_denied (bool):
        enterprise_status (EnterpriseStatus): Enterprise readiness status for endpoints.

            READY: Endpoint meets enterprise compliance standards (DPA, no training on data)
            PENDING: Endpoint is under review or does not yet meet enterprise standards
        reason (Union[Unset, str]):
        inherited (Union[Unset, bool]):  Default: False.
    """

    endpoint: str
    is_denied: bool
    enterprise_status: EnterpriseStatus
    reason: Union[Unset, str] = UNSET
    inherited: Union[Unset, bool] = False
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        endpoint = self.endpoint

        is_denied = self.is_denied

        enterprise_status = self.enterprise_status.value

        reason = self.reason

        inherited = self.inherited

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "endpoint": endpoint,
                "is_denied": is_denied,
                "enterprise_status": enterprise_status,
            }
        )
        if reason is not UNSET:
            field_dict["reason"] = reason
        if inherited is not UNSET:
            field_dict["inherited"] = inherited

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        endpoint = d.pop("endpoint")

        is_denied = d.pop("is_denied")

        enterprise_status = EnterpriseStatus(d.pop("enterprise_status"))

        reason = d.pop("reason", UNSET)

        inherited = d.pop("inherited", UNSET)

        endpoint_access_status = cls(
            endpoint=endpoint,
            is_denied=is_denied,
            enterprise_status=enterprise_status,
            reason=reason,
            inherited=inherited,
        )

        endpoint_access_status.additional_properties = d
        return endpoint_access_status

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

from typing import Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="LogDrainCreate")


@_attrs_define
class LogDrainCreate:
    """
    Attributes:
        name (str):
        endpoint_url (str):
        secret_token (str):
        sampling_rate (Union[Unset, int]):  Default: 1000.
    """

    name: str
    endpoint_url: str
    secret_token: str
    sampling_rate: Union[Unset, int] = 1000
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        name = self.name

        endpoint_url = self.endpoint_url

        secret_token = self.secret_token

        sampling_rate = self.sampling_rate

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "name": name,
                "endpoint_url": endpoint_url,
                "secret_token": secret_token,
            }
        )
        if sampling_rate is not UNSET:
            field_dict["sampling_rate"] = sampling_rate

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        name = d.pop("name")

        endpoint_url = d.pop("endpoint_url")

        secret_token = d.pop("secret_token")

        sampling_rate = d.pop("sampling_rate", UNSET)

        log_drain_create = cls(
            name=name,
            endpoint_url=endpoint_url,
            secret_token=secret_token,
            sampling_rate=sampling_rate,
        )

        log_drain_create.additional_properties = d
        return log_drain_create

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

from typing import Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="LogDrainUpdate")


@_attrs_define
class LogDrainUpdate:
    """
    Attributes:
        name (Union[Unset, str]):
        endpoint_url (Union[Unset, str]):
        secret_token (Union[Unset, str]):
        sampling_rate (Union[Unset, int]):
        is_active (Union[Unset, bool]):
    """

    name: Union[Unset, str] = UNSET
    endpoint_url: Union[Unset, str] = UNSET
    secret_token: Union[Unset, str] = UNSET
    sampling_rate: Union[Unset, int] = UNSET
    is_active: Union[Unset, bool] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        name = self.name

        endpoint_url = self.endpoint_url

        secret_token = self.secret_token

        sampling_rate = self.sampling_rate

        is_active = self.is_active

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if name is not UNSET:
            field_dict["name"] = name
        if endpoint_url is not UNSET:
            field_dict["endpoint_url"] = endpoint_url
        if secret_token is not UNSET:
            field_dict["secret_token"] = secret_token
        if sampling_rate is not UNSET:
            field_dict["sampling_rate"] = sampling_rate
        if is_active is not UNSET:
            field_dict["is_active"] = is_active

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        name = d.pop("name", UNSET)

        endpoint_url = d.pop("endpoint_url", UNSET)

        secret_token = d.pop("secret_token", UNSET)

        sampling_rate = d.pop("sampling_rate", UNSET)

        is_active = d.pop("is_active", UNSET)

        log_drain_update = cls(
            name=name,
            endpoint_url=endpoint_url,
            secret_token=secret_token,
            sampling_rate=sampling_rate,
            is_active=is_active,
        )

        log_drain_update.additional_properties = d
        return log_drain_update

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

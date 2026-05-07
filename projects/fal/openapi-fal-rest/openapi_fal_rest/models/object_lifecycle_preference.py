from typing import TYPE_CHECKING, Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.cdn_object_acl import CDNObjectACL


T = TypeVar("T", bound="ObjectLifecyclePreference")


@_attrs_define
class ObjectLifecyclePreference:
    """
    Attributes:
        allow_io_storage (Union[Unset, bool]):
        expiration_duration_seconds (Union[Unset, int]):
        initial_acl (Union[Unset, CDNObjectACL]):
    """

    allow_io_storage: Union[Unset, bool] = UNSET
    expiration_duration_seconds: Union[Unset, int] = UNSET
    initial_acl: Union[Unset, "CDNObjectACL"] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        allow_io_storage = self.allow_io_storage

        expiration_duration_seconds = self.expiration_duration_seconds

        initial_acl: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.initial_acl, Unset):
            initial_acl = self.initial_acl.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if allow_io_storage is not UNSET:
            field_dict["allow_io_storage"] = allow_io_storage
        if expiration_duration_seconds is not UNSET:
            field_dict["expiration_duration_seconds"] = expiration_duration_seconds
        if initial_acl is not UNSET:
            field_dict["initial_acl"] = initial_acl

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        from ..models.cdn_object_acl import CDNObjectACL

        d = src_dict.copy()
        allow_io_storage = d.pop("allow_io_storage", UNSET)

        expiration_duration_seconds = d.pop("expiration_duration_seconds", UNSET)

        _initial_acl = d.pop("initial_acl", UNSET)
        initial_acl: Union[Unset, CDNObjectACL]
        if isinstance(_initial_acl, Unset):
            initial_acl = UNSET
        else:
            initial_acl = CDNObjectACL.from_dict(_initial_acl)

        object_lifecycle_preference = cls(
            allow_io_storage=allow_io_storage,
            expiration_duration_seconds=expiration_duration_seconds,
            initial_acl=initial_acl,
        )

        object_lifecycle_preference.additional_properties = d
        return object_lifecycle_preference

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

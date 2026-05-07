from typing import TYPE_CHECKING, Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.body_create_realtime_token_extra_data import BodyCreateRealtimeTokenExtraData


T = TypeVar("T", bound="BodyCreateRealtimeToken")


@_attrs_define
class BodyCreateRealtimeToken:
    """
    Attributes:
        app (str): App identifier: 'app', 'owner/app', or 'owner/app/endpoint-path'
        token_expiration (Union[Unset, int]):  Default: 120.
        extra_data (Union[Unset, BodyCreateRealtimeTokenExtraData]):
    """

    app: str
    token_expiration: Union[Unset, int] = 120
    extra_data: Union[Unset, "BodyCreateRealtimeTokenExtraData"] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        app = self.app

        token_expiration = self.token_expiration

        extra_data: Union[Unset, dict[str, Any]] = UNSET
        if not isinstance(self.extra_data, Unset):
            extra_data = self.extra_data.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "app": app,
            }
        )
        if token_expiration is not UNSET:
            field_dict["token_expiration"] = token_expiration
        if extra_data is not UNSET:
            field_dict["extra_data"] = extra_data

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        from ..models.body_create_realtime_token_extra_data import BodyCreateRealtimeTokenExtraData

        d = src_dict.copy()
        app = d.pop("app")

        token_expiration = d.pop("token_expiration", UNSET)

        _extra_data = d.pop("extra_data", UNSET)
        extra_data: Union[Unset, BodyCreateRealtimeTokenExtraData]
        if isinstance(_extra_data, Unset):
            extra_data = UNSET
        else:
            extra_data = BodyCreateRealtimeTokenExtraData.from_dict(_extra_data)

        body_create_realtime_token = cls(
            app=app,
            token_expiration=token_expiration,
            extra_data=extra_data,
        )

        body_create_realtime_token.additional_properties = d
        return body_create_realtime_token

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

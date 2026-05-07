from typing import TYPE_CHECKING, Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

if TYPE_CHECKING:
    from ..models.key_auth_method_info import KeyAuthMethodInfo
    from ..models.user_auth_method_info import UserAuthMethodInfo


T = TypeVar("T", bound="DecorateAuthMethodsResponseDecorations")


@_attrs_define
class DecorateAuthMethodsResponseDecorations:
    """ """

    additional_properties: dict[str, Union["KeyAuthMethodInfo", "UserAuthMethodInfo"]] = _attrs_field(
        init=False, factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        from ..models.user_auth_method_info import UserAuthMethodInfo

        field_dict: dict[str, Any] = {}
        for prop_name, prop in self.additional_properties.items():
            if isinstance(prop, UserAuthMethodInfo):
                field_dict[prop_name] = prop.to_dict()
            else:
                field_dict[prop_name] = prop.to_dict()

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        from ..models.key_auth_method_info import KeyAuthMethodInfo
        from ..models.user_auth_method_info import UserAuthMethodInfo

        d = src_dict.copy()
        decorate_auth_methods_response_decorations = cls()

        additional_properties = {}
        for prop_name, prop_dict in d.items():

            def _parse_additional_property(data: object) -> Union["KeyAuthMethodInfo", "UserAuthMethodInfo"]:
                try:
                    if not isinstance(data, dict):
                        raise TypeError()
                    additional_property_type_0 = UserAuthMethodInfo.from_dict(data)

                    return additional_property_type_0
                except:  # noqa: E722
                    pass
                if not isinstance(data, dict):
                    raise TypeError()
                additional_property_type_1 = KeyAuthMethodInfo.from_dict(data)

                return additional_property_type_1

            additional_property = _parse_additional_property(prop_dict)

            additional_properties[prop_name] = additional_property

        decorate_auth_methods_response_decorations.additional_properties = additional_properties
        return decorate_auth_methods_response_decorations

    @property
    def additional_keys(self) -> list[str]:
        return list(self.additional_properties.keys())

    def __getitem__(self, key: str) -> Union["KeyAuthMethodInfo", "UserAuthMethodInfo"]:
        return self.additional_properties[key]

    def __setitem__(self, key: str, value: Union["KeyAuthMethodInfo", "UserAuthMethodInfo"]) -> None:
        self.additional_properties[key] = value

    def __delitem__(self, key: str) -> None:
        del self.additional_properties[key]

    def __contains__(self, key: str) -> bool:
        return key in self.additional_properties

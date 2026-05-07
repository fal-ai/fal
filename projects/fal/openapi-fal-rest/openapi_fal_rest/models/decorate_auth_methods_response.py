from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

if TYPE_CHECKING:
    from ..models.decorate_auth_methods_response_decorations import DecorateAuthMethodsResponseDecorations


T = TypeVar("T", bound="DecorateAuthMethodsResponse")


@_attrs_define
class DecorateAuthMethodsResponse:
    """Response containing decorated auth method information.

    Attributes:
        decorations (DecorateAuthMethodsResponseDecorations):
    """

    decorations: "DecorateAuthMethodsResponseDecorations"
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        decorations = self.decorations.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "decorations": decorations,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        from ..models.decorate_auth_methods_response_decorations import DecorateAuthMethodsResponseDecorations

        d = src_dict.copy()
        decorations = DecorateAuthMethodsResponseDecorations.from_dict(d.pop("decorations"))

        decorate_auth_methods_response = cls(
            decorations=decorations,
        )

        decorate_auth_methods_response.additional_properties = d
        return decorate_auth_methods_response

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

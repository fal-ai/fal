from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

if TYPE_CHECKING:
    from ..models.environment_info import EnvironmentInfo


T = TypeVar("T", bound="CreateEnvironmentResponse")


@_attrs_define
class CreateEnvironmentResponse:
    """Response for creating an environment.

    Attributes:
        environment (EnvironmentInfo): Environment information returned by the API.
    """

    environment: "EnvironmentInfo"
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        environment = self.environment.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "environment": environment,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        from ..models.environment_info import EnvironmentInfo

        d = src_dict.copy()
        environment = EnvironmentInfo.from_dict(d.pop("environment"))

        create_environment_response = cls(
            environment=environment,
        )

        create_environment_response.additional_properties = d
        return create_environment_response

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

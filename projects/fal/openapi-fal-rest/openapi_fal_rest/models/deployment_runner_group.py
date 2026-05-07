from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

T = TypeVar("T", bound="DeploymentRunnerGroup")


@_attrs_define
class DeploymentRunnerGroup:
    """
    Attributes:
        app_id (str):
        app_alias (str):
        num_runners (int):
    """

    app_id: str
    app_alias: str
    num_runners: int
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        app_id = self.app_id

        app_alias = self.app_alias

        num_runners = self.num_runners

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "app_id": app_id,
                "app_alias": app_alias,
                "num_runners": num_runners,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        app_id = d.pop("app_id")

        app_alias = d.pop("app_alias")

        num_runners = d.pop("num_runners")

        deployment_runner_group = cls(
            app_id=app_id,
            app_alias=app_alias,
            num_runners=num_runners,
        )

        deployment_runner_group.additional_properties = d
        return deployment_runner_group

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

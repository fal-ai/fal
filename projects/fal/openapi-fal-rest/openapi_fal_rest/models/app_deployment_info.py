from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

if TYPE_CHECKING:
    from ..models.deployment_runner_group import DeploymentRunnerGroup


T = TypeVar("T", bound="AppDeploymentInfo")


@_attrs_define
class AppDeploymentInfo:
    """
    Attributes:
        app_alias (str):
        rolling (bool):
        runner_groups (list['DeploymentRunnerGroup']):
    """

    app_alias: str
    rolling: bool
    runner_groups: list["DeploymentRunnerGroup"]
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        app_alias = self.app_alias

        rolling = self.rolling

        runner_groups = []
        for runner_groups_item_data in self.runner_groups:
            runner_groups_item = runner_groups_item_data.to_dict()
            runner_groups.append(runner_groups_item)

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "app_alias": app_alias,
                "rolling": rolling,
                "runner_groups": runner_groups,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        from ..models.deployment_runner_group import DeploymentRunnerGroup

        d = src_dict.copy()
        app_alias = d.pop("app_alias")

        rolling = d.pop("rolling")

        runner_groups = []
        _runner_groups = d.pop("runner_groups")
        for runner_groups_item_data in _runner_groups:
            runner_groups_item = DeploymentRunnerGroup.from_dict(runner_groups_item_data)

            runner_groups.append(runner_groups_item)

        app_deployment_info = cls(
            app_alias=app_alias,
            rolling=rolling,
            runner_groups=runner_groups,
        )

        app_deployment_info.additional_properties = d
        return app_deployment_info

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

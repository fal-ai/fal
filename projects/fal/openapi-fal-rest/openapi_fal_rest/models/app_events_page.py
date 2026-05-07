from typing import TYPE_CHECKING, Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.config_changed_event import ConfigChangedEvent
    from ..models.deployment_ended_event import DeploymentEndedEvent
    from ..models.deployment_failed_event import DeploymentFailedEvent
    from ..models.deployment_recreate_applied_event import DeploymentRecreateAppliedEvent
    from ..models.deployment_rolling_ended_event import DeploymentRollingEndedEvent
    from ..models.deployment_rolling_failed_event import DeploymentRollingFailedEvent
    from ..models.deployment_rolling_started_event import DeploymentRollingStartedEvent
    from ..models.deployment_started_event import DeploymentStartedEvent
    from ..models.runner_docker_pull_event import RunnerDockerPullEvent
    from ..models.runner_draining_event import RunnerDrainingEvent
    from ..models.runner_failed_event import RunnerFailedEvent
    from ..models.runner_finished_event import RunnerFinishedEvent
    from ..models.runner_pending_event import RunnerPendingEvent
    from ..models.runner_setup_event import RunnerSetupEvent
    from ..models.runner_started_event import RunnerStartedEvent
    from ..models.runner_stopping_event import RunnerStoppingEvent


T = TypeVar("T", bound="AppEventsPage")


@_attrs_define
class AppEventsPage:
    """
    Attributes:
        items (list[Union['ConfigChangedEvent', 'DeploymentEndedEvent', 'DeploymentFailedEvent',
            'DeploymentRecreateAppliedEvent', 'DeploymentRollingEndedEvent', 'DeploymentRollingFailedEvent',
            'DeploymentRollingStartedEvent', 'DeploymentStartedEvent', 'RunnerDockerPullEvent', 'RunnerDrainingEvent',
            'RunnerFailedEvent', 'RunnerFinishedEvent', 'RunnerPendingEvent', 'RunnerSetupEvent', 'RunnerStartedEvent',
            'RunnerStoppingEvent']]):
        total (Union[Unset, int]):
        page (Union[Unset, int]):
        size (Union[Unset, int]):
        pages (Union[Unset, int]):
    """

    items: list[
        Union[
            "ConfigChangedEvent",
            "DeploymentEndedEvent",
            "DeploymentFailedEvent",
            "DeploymentRecreateAppliedEvent",
            "DeploymentRollingEndedEvent",
            "DeploymentRollingFailedEvent",
            "DeploymentRollingStartedEvent",
            "DeploymentStartedEvent",
            "RunnerDockerPullEvent",
            "RunnerDrainingEvent",
            "RunnerFailedEvent",
            "RunnerFinishedEvent",
            "RunnerPendingEvent",
            "RunnerSetupEvent",
            "RunnerStartedEvent",
            "RunnerStoppingEvent",
        ]
    ]
    total: Union[Unset, int] = UNSET
    page: Union[Unset, int] = UNSET
    size: Union[Unset, int] = UNSET
    pages: Union[Unset, int] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        from ..models.config_changed_event import ConfigChangedEvent
        from ..models.deployment_failed_event import DeploymentFailedEvent
        from ..models.deployment_recreate_applied_event import DeploymentRecreateAppliedEvent
        from ..models.deployment_rolling_ended_event import DeploymentRollingEndedEvent
        from ..models.deployment_rolling_failed_event import DeploymentRollingFailedEvent
        from ..models.deployment_rolling_started_event import DeploymentRollingStartedEvent
        from ..models.deployment_started_event import DeploymentStartedEvent
        from ..models.runner_docker_pull_event import RunnerDockerPullEvent
        from ..models.runner_draining_event import RunnerDrainingEvent
        from ..models.runner_failed_event import RunnerFailedEvent
        from ..models.runner_finished_event import RunnerFinishedEvent
        from ..models.runner_pending_event import RunnerPendingEvent
        from ..models.runner_setup_event import RunnerSetupEvent
        from ..models.runner_started_event import RunnerStartedEvent
        from ..models.runner_stopping_event import RunnerStoppingEvent

        items = []
        for items_item_data in self.items:
            items_item: dict[str, Any]
            if isinstance(items_item_data, RunnerStartedEvent):
                items_item = items_item_data.to_dict()
            elif isinstance(items_item_data, RunnerFailedEvent):
                items_item = items_item_data.to_dict()
            elif isinstance(items_item_data, RunnerFinishedEvent):
                items_item = items_item_data.to_dict()
            elif isinstance(items_item_data, RunnerPendingEvent):
                items_item = items_item_data.to_dict()
            elif isinstance(items_item_data, RunnerDockerPullEvent):
                items_item = items_item_data.to_dict()
            elif isinstance(items_item_data, RunnerSetupEvent):
                items_item = items_item_data.to_dict()
            elif isinstance(items_item_data, RunnerDrainingEvent):
                items_item = items_item_data.to_dict()
            elif isinstance(items_item_data, RunnerStoppingEvent):
                items_item = items_item_data.to_dict()
            elif isinstance(items_item_data, DeploymentRollingStartedEvent):
                items_item = items_item_data.to_dict()
            elif isinstance(items_item_data, DeploymentRollingFailedEvent):
                items_item = items_item_data.to_dict()
            elif isinstance(items_item_data, DeploymentRollingEndedEvent):
                items_item = items_item_data.to_dict()
            elif isinstance(items_item_data, DeploymentRecreateAppliedEvent):
                items_item = items_item_data.to_dict()
            elif isinstance(items_item_data, ConfigChangedEvent):
                items_item = items_item_data.to_dict()
            elif isinstance(items_item_data, DeploymentStartedEvent):
                items_item = items_item_data.to_dict()
            elif isinstance(items_item_data, DeploymentFailedEvent):
                items_item = items_item_data.to_dict()
            else:
                items_item = items_item_data.to_dict()

            items.append(items_item)

        total = self.total

        page = self.page

        size = self.size

        pages = self.pages

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "items": items,
            }
        )
        if total is not UNSET:
            field_dict["total"] = total
        if page is not UNSET:
            field_dict["page"] = page
        if size is not UNSET:
            field_dict["size"] = size
        if pages is not UNSET:
            field_dict["pages"] = pages

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        from ..models.config_changed_event import ConfigChangedEvent
        from ..models.deployment_ended_event import DeploymentEndedEvent
        from ..models.deployment_failed_event import DeploymentFailedEvent
        from ..models.deployment_recreate_applied_event import DeploymentRecreateAppliedEvent
        from ..models.deployment_rolling_ended_event import DeploymentRollingEndedEvent
        from ..models.deployment_rolling_failed_event import DeploymentRollingFailedEvent
        from ..models.deployment_rolling_started_event import DeploymentRollingStartedEvent
        from ..models.deployment_started_event import DeploymentStartedEvent
        from ..models.runner_docker_pull_event import RunnerDockerPullEvent
        from ..models.runner_draining_event import RunnerDrainingEvent
        from ..models.runner_failed_event import RunnerFailedEvent
        from ..models.runner_finished_event import RunnerFinishedEvent
        from ..models.runner_pending_event import RunnerPendingEvent
        from ..models.runner_setup_event import RunnerSetupEvent
        from ..models.runner_started_event import RunnerStartedEvent
        from ..models.runner_stopping_event import RunnerStoppingEvent

        d = src_dict.copy()
        items = []
        _items = d.pop("items")
        for items_item_data in _items:

            def _parse_items_item(
                data: object,
            ) -> Union[
                "ConfigChangedEvent",
                "DeploymentEndedEvent",
                "DeploymentFailedEvent",
                "DeploymentRecreateAppliedEvent",
                "DeploymentRollingEndedEvent",
                "DeploymentRollingFailedEvent",
                "DeploymentRollingStartedEvent",
                "DeploymentStartedEvent",
                "RunnerDockerPullEvent",
                "RunnerDrainingEvent",
                "RunnerFailedEvent",
                "RunnerFinishedEvent",
                "RunnerPendingEvent",
                "RunnerSetupEvent",
                "RunnerStartedEvent",
                "RunnerStoppingEvent",
            ]:
                try:
                    if not isinstance(data, dict):
                        raise TypeError()
                    items_item_type_0 = RunnerStartedEvent.from_dict(data)

                    return items_item_type_0
                except:  # noqa: E722
                    pass
                try:
                    if not isinstance(data, dict):
                        raise TypeError()
                    items_item_type_1 = RunnerFailedEvent.from_dict(data)

                    return items_item_type_1
                except:  # noqa: E722
                    pass
                try:
                    if not isinstance(data, dict):
                        raise TypeError()
                    items_item_type_2 = RunnerFinishedEvent.from_dict(data)

                    return items_item_type_2
                except:  # noqa: E722
                    pass
                try:
                    if not isinstance(data, dict):
                        raise TypeError()
                    items_item_type_3 = RunnerPendingEvent.from_dict(data)

                    return items_item_type_3
                except:  # noqa: E722
                    pass
                try:
                    if not isinstance(data, dict):
                        raise TypeError()
                    items_item_type_4 = RunnerDockerPullEvent.from_dict(data)

                    return items_item_type_4
                except:  # noqa: E722
                    pass
                try:
                    if not isinstance(data, dict):
                        raise TypeError()
                    items_item_type_5 = RunnerSetupEvent.from_dict(data)

                    return items_item_type_5
                except:  # noqa: E722
                    pass
                try:
                    if not isinstance(data, dict):
                        raise TypeError()
                    items_item_type_6 = RunnerDrainingEvent.from_dict(data)

                    return items_item_type_6
                except:  # noqa: E722
                    pass
                try:
                    if not isinstance(data, dict):
                        raise TypeError()
                    items_item_type_7 = RunnerStoppingEvent.from_dict(data)

                    return items_item_type_7
                except:  # noqa: E722
                    pass
                try:
                    if not isinstance(data, dict):
                        raise TypeError()
                    items_item_type_8 = DeploymentRollingStartedEvent.from_dict(data)

                    return items_item_type_8
                except:  # noqa: E722
                    pass
                try:
                    if not isinstance(data, dict):
                        raise TypeError()
                    items_item_type_9 = DeploymentRollingFailedEvent.from_dict(data)

                    return items_item_type_9
                except:  # noqa: E722
                    pass
                try:
                    if not isinstance(data, dict):
                        raise TypeError()
                    items_item_type_10 = DeploymentRollingEndedEvent.from_dict(data)

                    return items_item_type_10
                except:  # noqa: E722
                    pass
                try:
                    if not isinstance(data, dict):
                        raise TypeError()
                    items_item_type_11 = DeploymentRecreateAppliedEvent.from_dict(data)

                    return items_item_type_11
                except:  # noqa: E722
                    pass
                try:
                    if not isinstance(data, dict):
                        raise TypeError()
                    items_item_type_12 = ConfigChangedEvent.from_dict(data)

                    return items_item_type_12
                except:  # noqa: E722
                    pass
                try:
                    if not isinstance(data, dict):
                        raise TypeError()
                    items_item_type_13 = DeploymentStartedEvent.from_dict(data)

                    return items_item_type_13
                except:  # noqa: E722
                    pass
                try:
                    if not isinstance(data, dict):
                        raise TypeError()
                    items_item_type_14 = DeploymentFailedEvent.from_dict(data)

                    return items_item_type_14
                except:  # noqa: E722
                    pass
                if not isinstance(data, dict):
                    raise TypeError()
                items_item_type_15 = DeploymentEndedEvent.from_dict(data)

                return items_item_type_15

            items_item = _parse_items_item(items_item_data)

            items.append(items_item)

        total = d.pop("total", UNSET)

        page = d.pop("page", UNSET)

        size = d.pop("size", UNSET)

        pages = d.pop("pages", UNSET)

        app_events_page = cls(
            items=items,
            total=total,
            page=page,
            size=size,
            pages=pages,
        )

        app_events_page.additional_properties = d
        return app_events_page

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

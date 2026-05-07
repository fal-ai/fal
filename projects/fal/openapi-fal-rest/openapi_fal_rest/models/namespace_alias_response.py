import datetime
from typing import Any, TypeVar, Union
from uuid import UUID

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from dateutil.parser import isoparse

from ..types import UNSET, Unset

T = TypeVar("T", bound="NamespaceAliasResponse")


@_attrs_define
class NamespaceAliasResponse:
    """
    Attributes:
        id (UUID):
        source_user_nickname (str):
        source_app_name (str):
        source_path_pattern (str):
        target_user_nickname (str):
        target_app_name (str):
        created_by_nickname (str):
        created_at (datetime.datetime):
        active (bool):
        target_path_template (Union[Unset, str]):
    """

    id: UUID
    source_user_nickname: str
    source_app_name: str
    source_path_pattern: str
    target_user_nickname: str
    target_app_name: str
    created_by_nickname: str
    created_at: datetime.datetime
    active: bool
    target_path_template: Union[Unset, str] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        id = str(self.id)

        source_user_nickname = self.source_user_nickname

        source_app_name = self.source_app_name

        source_path_pattern = self.source_path_pattern

        target_user_nickname = self.target_user_nickname

        target_app_name = self.target_app_name

        created_by_nickname = self.created_by_nickname

        created_at = self.created_at.isoformat()

        active = self.active

        target_path_template = self.target_path_template

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "id": id,
                "source_user_nickname": source_user_nickname,
                "source_app_name": source_app_name,
                "source_path_pattern": source_path_pattern,
                "target_user_nickname": target_user_nickname,
                "target_app_name": target_app_name,
                "created_by_nickname": created_by_nickname,
                "created_at": created_at,
                "active": active,
            }
        )
        if target_path_template is not UNSET:
            field_dict["target_path_template"] = target_path_template

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        id = UUID(d.pop("id"))

        source_user_nickname = d.pop("source_user_nickname")

        source_app_name = d.pop("source_app_name")

        source_path_pattern = d.pop("source_path_pattern")

        target_user_nickname = d.pop("target_user_nickname")

        target_app_name = d.pop("target_app_name")

        created_by_nickname = d.pop("created_by_nickname")

        created_at = isoparse(d.pop("created_at"))

        active = d.pop("active")

        target_path_template = d.pop("target_path_template", UNSET)

        namespace_alias_response = cls(
            id=id,
            source_user_nickname=source_user_nickname,
            source_app_name=source_app_name,
            source_path_pattern=source_path_pattern,
            target_user_nickname=target_user_nickname,
            target_app_name=target_app_name,
            created_by_nickname=created_by_nickname,
            created_at=created_at,
            active=active,
            target_path_template=target_path_template,
        )

        namespace_alias_response.additional_properties = d
        return namespace_alias_response

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

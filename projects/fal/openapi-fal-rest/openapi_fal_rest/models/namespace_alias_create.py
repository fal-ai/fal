from typing import Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="NamespaceAliasCreate")


@_attrs_define
class NamespaceAliasCreate:
    """
    Attributes:
        source_user_nickname (str): Source user nickname (can be vanity)
        source_app (str): Source app name
        target_user_nickname (str): Target user nickname
        target_app (str): Target app name
        source_path (Union[Unset, str]): Source path pattern (empty, specific, or '*' for wildcard) Default: ''.
        target_path_template (Union[Unset, str]): Target path template using {path} placeholder
    """

    source_user_nickname: str
    source_app: str
    target_user_nickname: str
    target_app: str
    source_path: Union[Unset, str] = ""
    target_path_template: Union[Unset, str] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        source_user_nickname = self.source_user_nickname

        source_app = self.source_app

        target_user_nickname = self.target_user_nickname

        target_app = self.target_app

        source_path = self.source_path

        target_path_template = self.target_path_template

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "source_user_nickname": source_user_nickname,
                "source_app": source_app,
                "target_user_nickname": target_user_nickname,
                "target_app": target_app,
            }
        )
        if source_path is not UNSET:
            field_dict["source_path"] = source_path
        if target_path_template is not UNSET:
            field_dict["target_path_template"] = target_path_template

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        source_user_nickname = d.pop("source_user_nickname")

        source_app = d.pop("source_app")

        target_user_nickname = d.pop("target_user_nickname")

        target_app = d.pop("target_app")

        source_path = d.pop("source_path", UNSET)

        target_path_template = d.pop("target_path_template", UNSET)

        namespace_alias_create = cls(
            source_user_nickname=source_user_nickname,
            source_app=source_app,
            target_user_nickname=target_user_nickname,
            target_app=target_app,
            source_path=source_path,
            target_path_template=target_path_template,
        )

        namespace_alias_create.additional_properties = d
        return namespace_alias_create

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

from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.cdn_object_acl_decision import CDNObjectACLDecision

if TYPE_CHECKING:
    from ..models.cdn_object_acl_rule import CDNObjectACLRule


T = TypeVar("T", bound="CDNObjectACL")


@_attrs_define
class CDNObjectACL:
    """
    Attributes:
        default (CDNObjectACLDecision): Access control decision for fal CDN objects.

            Determines what happens when a user attempts to access a CDN object:
            - HIDE: The object appears to not exist (returns 404)
            - FORBID: Access is explicitly denied (returns 403)
            - ALLOW: Access is granted
        rules (list['CDNObjectACLRule']):
    """

    default: CDNObjectACLDecision
    rules: list["CDNObjectACLRule"]
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        default = self.default.value

        rules = []
        for rules_item_data in self.rules:
            rules_item = rules_item_data.to_dict()
            rules.append(rules_item)

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "default": default,
                "rules": rules,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        from ..models.cdn_object_acl_rule import CDNObjectACLRule

        d = src_dict.copy()
        default = CDNObjectACLDecision(d.pop("default"))

        rules = []
        _rules = d.pop("rules")
        for rules_item_data in _rules:
            rules_item = CDNObjectACLRule.from_dict(rules_item_data)

            rules.append(rules_item)

        cdn_object_acl = cls(
            default=default,
            rules=rules,
        )

        cdn_object_acl.additional_properties = d
        return cdn_object_acl

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

from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.cdn_object_acl_decision import CDNObjectACLDecision

T = TypeVar("T", bound="CdnObjectAclRuleData")


@_attrs_define
class CdnObjectAclRuleData:
    """
    Attributes:
        user (str):
        decision (CDNObjectACLDecision): Access control decision for fal CDN objects.

            Determines what happens when a user attempts to access a CDN object:
            - HIDE: The object appears to not exist (returns 404)
            - FORBID: Access is explicitly denied (returns 403)
            - ALLOW: Access is granted
    """

    user: str
    decision: CDNObjectACLDecision
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        user = self.user

        decision = self.decision.value

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "user": user,
                "decision": decision,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        d = src_dict.copy()
        user = d.pop("user")

        decision = CDNObjectACLDecision(d.pop("decision"))

        cdn_object_acl_rule_data = cls(
            user=user,
            decision=decision,
        )

        cdn_object_acl_rule_data.additional_properties = d
        return cdn_object_acl_rule_data

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

from typing import TYPE_CHECKING, Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.cdn_object_acl_decision import CDNObjectACLDecision
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.cdn_object_acl_rule_data import CdnObjectAclRuleData


T = TypeVar("T", bound="CdnObjectAclData")


@_attrs_define
class CdnObjectAclData:
    """
    Attributes:
        default (CDNObjectACLDecision): Access control decision for fal CDN objects.

            Determines what happens when a user attempts to access a CDN object:
            - HIDE: The object appears to not exist (returns 404)
            - FORBID: Access is explicitly denied (returns 403)
            - ALLOW: Access is granted
        rules (Union[Unset, list['CdnObjectAclRuleData']]):
    """

    default: CDNObjectACLDecision
    rules: Union[Unset, list["CdnObjectAclRuleData"]] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        default = self.default.value

        rules: Union[Unset, list[dict[str, Any]]] = UNSET
        if not isinstance(self.rules, Unset):
            rules = []
            for rules_item_data in self.rules:
                rules_item = rules_item_data.to_dict()
                rules.append(rules_item)

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "default": default,
            }
        )
        if rules is not UNSET:
            field_dict["rules"] = rules

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        from ..models.cdn_object_acl_rule_data import CdnObjectAclRuleData

        d = src_dict.copy()
        default = CDNObjectACLDecision(d.pop("default"))

        rules = []
        _rules = d.pop("rules", UNSET)
        for rules_item_data in _rules or []:
            rules_item = CdnObjectAclRuleData.from_dict(rules_item_data)

            rules.append(rules_item)

        cdn_object_acl_data = cls(
            default=default,
            rules=rules,
        )

        cdn_object_acl_data.additional_properties = d
        return cdn_object_acl_data

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

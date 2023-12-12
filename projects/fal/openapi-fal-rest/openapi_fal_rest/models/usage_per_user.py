from typing import Any, Dict, List, Type, TypeVar

import attr

T = TypeVar("T", bound="UsagePerUser")


@attr.s(auto_attribs=True)
class UsagePerUser:
    """
    Attributes:
        user_id (str):
        machine_type (str):
        total_billable_duration (int):
    """

    user_id: str
    machine_type: str
    total_billable_duration: int
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        user_id = self.user_id
        machine_type = self.machine_type
        total_billable_duration = self.total_billable_duration

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "user_id": user_id,
                "machine_type": machine_type,
                "total_billable_duration": total_billable_duration,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        user_id = d.pop("user_id")

        machine_type = d.pop("machine_type")

        total_billable_duration = d.pop("total_billable_duration")

        usage_per_user = cls(
            user_id=user_id,
            machine_type=machine_type,
            total_billable_duration=total_billable_duration,
        )

        usage_per_user.additional_properties = d
        return usage_per_user

    @property
    def additional_keys(self) -> List[str]:
        return list(self.additional_properties.keys())

    def __getitem__(self, key: str) -> Any:
        return self.additional_properties[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.additional_properties[key] = value

    def __delitem__(self, key: str) -> None:
        del self.additional_properties[key]

    def __contains__(self, key: str) -> bool:
        return key in self.additional_properties

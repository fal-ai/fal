import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Type, TypeVar

import attr
from dateutil.parser import isoparse

from ..models.run_type import RunType

if TYPE_CHECKING:
    from ..models.persisted_usage_record_meta import PersistedUsageRecordMeta


T = TypeVar("T", bound="PersistedUsageRecord")


@attr.s(auto_attribs=True)
class PersistedUsageRecord:
    """
    Attributes:
        id (str):
        subscription_item_id (str):
        subscription_id (str):
        quantity (int):
        run_type (RunType): An enumeration.
        machine_type (str):
        timestamp (datetime.datetime):
        meta (PersistedUsageRecordMeta):
    """

    id: str
    subscription_item_id: str
    subscription_id: str
    quantity: int
    run_type: RunType
    machine_type: str
    timestamp: datetime.datetime
    meta: "PersistedUsageRecordMeta"
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        id = self.id
        subscription_item_id = self.subscription_item_id
        subscription_id = self.subscription_id
        quantity = self.quantity
        run_type = self.run_type.value

        machine_type = self.machine_type
        timestamp = self.timestamp.isoformat()

        meta = self.meta.to_dict()

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "id": id,
                "subscription_item_id": subscription_item_id,
                "subscription_id": subscription_id,
                "quantity": quantity,
                "run_type": run_type,
                "machine_type": machine_type,
                "timestamp": timestamp,
                "meta": meta,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        from ..models.persisted_usage_record_meta import PersistedUsageRecordMeta

        d = src_dict.copy()
        id = d.pop("id")

        subscription_item_id = d.pop("subscription_item_id")

        subscription_id = d.pop("subscription_id")

        quantity = d.pop("quantity")

        run_type = RunType(d.pop("run_type"))

        machine_type = d.pop("machine_type")

        timestamp = isoparse(d.pop("timestamp"))

        meta = PersistedUsageRecordMeta.from_dict(d.pop("meta"))

        persisted_usage_record = cls(
            id=id,
            subscription_item_id=subscription_item_id,
            subscription_id=subscription_id,
            quantity=quantity,
            run_type=run_type,
            machine_type=machine_type,
            timestamp=timestamp,
            meta=meta,
        )

        persisted_usage_record.additional_properties = d
        return persisted_usage_record

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

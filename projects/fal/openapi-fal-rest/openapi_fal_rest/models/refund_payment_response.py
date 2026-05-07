from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

if TYPE_CHECKING:
    from ..models.vercel_balance import VercelBalance


T = TypeVar("T", bound="RefundPaymentResponse")


@_attrs_define
class RefundPaymentResponse:
    """
    Attributes:
        timestamp (str):
        balances (list['VercelBalance']):
    """

    timestamp: str
    balances: list["VercelBalance"]
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        timestamp = self.timestamp

        balances = []
        for balances_item_data in self.balances:
            balances_item = balances_item_data.to_dict()
            balances.append(balances_item)

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "timestamp": timestamp,
                "balances": balances,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        from ..models.vercel_balance import VercelBalance

        d = src_dict.copy()
        timestamp = d.pop("timestamp")

        balances = []
        _balances = d.pop("balances")
        for balances_item_data in _balances:
            balances_item = VercelBalance.from_dict(balances_item_data)

            balances.append(balances_item)

        refund_payment_response = cls(
            timestamp=timestamp,
            balances=balances,
        )

        refund_payment_response.additional_properties = d
        return refund_payment_response

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

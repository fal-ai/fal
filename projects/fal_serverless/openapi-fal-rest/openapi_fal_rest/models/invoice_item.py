from typing import Any, Dict, List, Type, TypeVar

import attr

T = TypeVar("T", bound="InvoiceItem")


@attr.s(auto_attribs=True)
class InvoiceItem:
    """
    Attributes:
        name (str):
        amount (int):
        quantity (int):
        machine_type (str):
        run_type (str):
    """

    name: str
    amount: int
    quantity: int
    machine_type: str
    run_type: str
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        name = self.name
        amount = self.amount
        quantity = self.quantity
        machine_type = self.machine_type
        run_type = self.run_type

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "name": name,
                "amount": amount,
                "quantity": quantity,
                "machine_type": machine_type,
                "run_type": run_type,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        name = d.pop("name")

        amount = d.pop("amount")

        quantity = d.pop("quantity")

        machine_type = d.pop("machine_type")

        run_type = d.pop("run_type")

        invoice_item = cls(
            name=name,
            amount=amount,
            quantity=quantity,
            machine_type=machine_type,
            run_type=run_type,
        )

        invoice_item.additional_properties = d
        return invoice_item

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

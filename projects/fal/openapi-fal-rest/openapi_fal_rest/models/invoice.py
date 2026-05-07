from typing import TYPE_CHECKING, Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.invoice_item import InvoiceItem


T = TypeVar("T", bound="Invoice")


@_attrs_define
class Invoice:
    """
    Attributes:
        amount_due (int):
        period_start (int):
        period_end (int):
        created (int):
        subtotal (int):
        total (int):
        items (list['InvoiceItem']):
        url (str):
        starting_balance (int):
        status (str):
        effective_at (Union[Unset, int]):
    """

    amount_due: int
    period_start: int
    period_end: int
    created: int
    subtotal: int
    total: int
    items: list["InvoiceItem"]
    url: str
    starting_balance: int
    status: str
    effective_at: Union[Unset, int] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        amount_due = self.amount_due

        period_start = self.period_start

        period_end = self.period_end

        created = self.created

        subtotal = self.subtotal

        total = self.total

        items = []
        for items_item_data in self.items:
            items_item = items_item_data.to_dict()
            items.append(items_item)

        url = self.url

        starting_balance = self.starting_balance

        status = self.status

        effective_at = self.effective_at

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "amount_due": amount_due,
                "period_start": period_start,
                "period_end": period_end,
                "created": created,
                "subtotal": subtotal,
                "total": total,
                "items": items,
                "url": url,
                "starting_balance": starting_balance,
                "status": status,
            }
        )
        if effective_at is not UNSET:
            field_dict["effective_at"] = effective_at

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: dict[str, Any]) -> T:
        from ..models.invoice_item import InvoiceItem

        d = src_dict.copy()
        amount_due = d.pop("amount_due")

        period_start = d.pop("period_start")

        period_end = d.pop("period_end")

        created = d.pop("created")

        subtotal = d.pop("subtotal")

        total = d.pop("total")

        items = []
        _items = d.pop("items")
        for items_item_data in _items:
            items_item = InvoiceItem.from_dict(items_item_data)

            items.append(items_item)

        url = d.pop("url")

        starting_balance = d.pop("starting_balance")

        status = d.pop("status")

        effective_at = d.pop("effective_at", UNSET)

        invoice = cls(
            amount_due=amount_due,
            period_start=period_start,
            period_end=period_end,
            created=created,
            subtotal=subtotal,
            total=total,
            items=items,
            url=url,
            starting_balance=starting_balance,
            status=status,
            effective_at=effective_at,
        )

        invoice.additional_properties = d
        return invoice

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

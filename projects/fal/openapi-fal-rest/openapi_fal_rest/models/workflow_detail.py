import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Type, TypeVar, Union, cast

import attr
from dateutil.parser import isoparse

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.workflow_detail_contents_type_0 import WorkflowDetailContentsType0


T = TypeVar("T", bound="WorkflowDetail")


@attr.s(auto_attribs=True)
class WorkflowDetail:
    """
    Attributes:
        name (str):
        title (str):
        is_public (bool):
        user_id (str):
        created_at (datetime.datetime):
        contents (Union['WorkflowDetailContentsType0', List[Any], Unset, bool, float, int, str]):
    """

    name: str
    title: str
    is_public: bool
    user_id: str
    created_at: datetime.datetime
    contents: Union["WorkflowDetailContentsType0", List[Any], Unset, bool, float, int, str] = UNSET
    additional_properties: Dict[str, Any] = attr.ib(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        from ..models.workflow_detail_contents_type_0 import WorkflowDetailContentsType0

        name = self.name
        title = self.title
        is_public = self.is_public
        user_id = self.user_id
        created_at = self.created_at.isoformat()

        contents: Union[Dict[str, Any], List[Any], Unset, bool, float, int, str]
        if isinstance(self.contents, Unset):
            contents = UNSET

        elif isinstance(self.contents, WorkflowDetailContentsType0):
            contents = UNSET
            if not isinstance(self.contents, Unset):
                contents = self.contents.to_dict()

        elif isinstance(self.contents, list):
            contents = UNSET
            if not isinstance(self.contents, Unset):
                contents = self.contents

        else:
            contents = self.contents

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "name": name,
                "title": title,
                "is_public": is_public,
                "user_id": user_id,
                "created_at": created_at,
            }
        )
        if contents is not UNSET:
            field_dict["contents"] = contents

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        from ..models.workflow_detail_contents_type_0 import WorkflowDetailContentsType0

        d = src_dict.copy()
        name = d.pop("name")

        title = d.pop("title")

        is_public = d.pop("is_public")

        user_id = d.pop("user_id")

        created_at = isoparse(d.pop("created_at"))

        def _parse_contents(
            data: object,
        ) -> Union["WorkflowDetailContentsType0", List[Any], Unset, bool, float, int, str]:
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                _contents_type_0 = data
                contents_type_0: Union[Unset, WorkflowDetailContentsType0]
                if isinstance(_contents_type_0, Unset):
                    contents_type_0 = UNSET
                else:
                    contents_type_0 = WorkflowDetailContentsType0.from_dict(_contents_type_0)

                return contents_type_0
            except:  # noqa: E722
                pass
            try:
                if not isinstance(data, list):
                    raise TypeError()
                contents_type_1 = cast(List[Any], data)

                return contents_type_1
            except:  # noqa: E722
                pass
            return cast(Union["WorkflowDetailContentsType0", List[Any], Unset, bool, float, int, str], data)

        contents = _parse_contents(d.pop("contents", UNSET))

        workflow_detail = cls(
            name=name,
            title=title,
            is_public=is_public,
            user_id=user_id,
            created_at=created_at,
            contents=contents,
        )

        workflow_detail.additional_properties = d
        return workflow_detail

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

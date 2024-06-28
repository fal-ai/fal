import pytest

from fal_client.client import Queued, InProgress, Completed, _BaseRequestHandle


@pytest.mark.parametrize(
    "data, result, raised",
    [
        (
            {"status": "IN_QUEUE", "queue_position": 123},
            Queued(position=123),
            False,
        ),
        (
            {"status": "IN_PROGRESS", "logs": [{"msg": "foo"}, {"msg": "bar"}]},
            InProgress(logs=[{"msg": "foo"}, {"msg": "bar"}]),
            False,
        ),
        (
            {"status": "COMPLETED", "logs": [{"msg": "foo"}, {"msg": "bar"}]},
            Completed(logs=[{"msg": "foo"}, {"msg": "bar"}], metrics={}),
            False,
        ),
        (
            {"status": "COMPLETED", "logs": [{"msg": "foo"}, {"msg": "bar"}], "metrics": {"m1": "v1", "m2": "v2"}},
            Completed(logs=[{"msg": "foo"}, {"msg": "bar"}], metrics={"m1": "v1", "m2": "v2"}),
            False,
        ),
        (
            {"status": "FOO"},
            ValueError,
            True,
        )
    ]
)
def test_parse_status(data, result, raised):
    handle = _BaseRequestHandle("foo", "bar", "baz", "qux")

    if raised:
        with pytest.raises(result):
            handle._parse_status(data)
    else:
        assert handle._parse_status(data) == result

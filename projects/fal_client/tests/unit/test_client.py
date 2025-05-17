import pytest
from datetime import datetime

from fal_client.client import (
    Queued,
    InProgress,
    Completed,
    _BaseRequestHandle,
    LogMessage,
)


@pytest.mark.parametrize(
    "data, result, raised",
    [
        (
            {"status": "IN_QUEUE", "queue_position": 123},
            Queued(position=123),
            False,
        ),
        (
            {"status": "IN_PROGRESS", "logs": [{"message": "foo"}, {"message": "bar"}]},
            InProgress(
                logs=[
                    LogMessage(
                        timestamp=datetime.now(),
                        level="INFO",
                        message="foo",
                        source="unknown",
                    ),
                    LogMessage(
                        timestamp=datetime.now(),
                        level="INFO",
                        message="bar",
                        source="unknown",
                    ),
                ]
            ),
            False,
        ),
        (
            {"status": "COMPLETED", "logs": [{"message": "foo"}, {"message": "bar"}]},
            Completed(
                logs=[
                    LogMessage(
                        timestamp=datetime.now(),
                        level="INFO",
                        message="foo",
                        source="unknown",
                    ),
                    LogMessage(
                        timestamp=datetime.now(),
                        level="INFO",
                        message="bar",
                        source="unknown",
                    ),
                ],
                metrics={},
            ),
            False,
        ),
        (
            {
                "status": "COMPLETED",
                "logs": [{"message": "foo"}, {"message": "bar"}],
                "metrics": {"m1": "v1", "m2": "v2"},
            },
            Completed(
                logs=[
                    LogMessage(
                        timestamp=datetime.now(),
                        level="INFO",
                        message="foo",
                        source="unknown",
                    ),
                    LogMessage(
                        timestamp=datetime.now(),
                        level="INFO",
                        message="bar",
                        source="unknown",
                    ),
                ],
                metrics={"m1": "v1", "m2": "v2"},
            ),
            False,
        ),
        (
            {"status": "FOO"},
            ValueError,
            True,
        ),
    ],
)
def test_parse_status(data, result, raised):
    handle = _BaseRequestHandle("foo", "bar", "baz", "qux")

    if raised:
        with pytest.raises(result):
            handle._parse_status(data)
    else:
        actual = handle._parse_status(data)
        if isinstance(actual, (InProgress, Completed)) and actual.logs:
            # For log messages, we only compare the message field since timestamp will be different
            assert len(actual.logs) == len(result.logs)
            for actual_log, expected_log in zip(actual.logs, result.logs):
                assert actual_log.message == expected_log.message
                assert actual_log.level == expected_log.level
                assert actual_log.source == expected_log.source
            # Compare metrics separately for Completed status
            if isinstance(actual, Completed):
                assert actual.metrics == result.metrics
        else:
            assert actual == result

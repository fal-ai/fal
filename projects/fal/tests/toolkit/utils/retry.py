from unittest import mock

import pytest
from fal.toolkit.utils.retry import retry


def test_retry_success():
    @retry(max_retries=3, base_delay=1, backoff_type="fixed")
    def successful_function():
        return "Success"

    assert successful_function() == "Success"


def test_retry_failure():
    @retry(max_retries=3, base_delay=1, backoff_type="fixed")
    def failing_function():
        raise ValueError("Test error")

    with pytest.raises(ValueError, match="Test error"):
        failing_function()


@mock.patch("time.sleep")
def test_retry_exponential(mock_sleep):
    attempts = 0

    @retry(max_retries=3, base_delay=1, max_delay=60, backoff_type="exponential")
    def fail_twice_then_succeed():
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise ValueError("Test error")
        return "Success"

    assert fail_twice_then_succeed() == "Success"
    assert attempts == 3
    mock_sleep.assert_has_calls(
        [
            mock.call(1),
            mock.call(2),
        ]
    )


@mock.patch("time.sleep")
def test_retry_fixed(mock_sleep):
    attempts = 0

    @retry(max_retries=3, base_delay=2, backoff_type="fixed")
    def fail_twice_then_succeed():
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise ValueError("Test error")
        return "Success"

    assert fail_twice_then_succeed() == "Success"
    assert attempts == 3
    mock_sleep.assert_has_calls(
        [
            mock.call(2),
            mock.call(2),
        ]
    )


@mock.patch("random.uniform", return_value=1.25)
@mock.patch("time.sleep")
def test_retry_jitter(mock_sleep, mock_uniform):
    attempts = 0

    @retry(max_retries=3, base_delay=1, backoff_type="fixed", jitter=True)
    def fail_twice_then_succeed():
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise ValueError("Test error")
        return "Success"

    assert fail_twice_then_succeed() == "Success"
    assert attempts == 3
    mock_sleep.assert_has_calls(
        [
            mock.call(1.25),
            mock.call(1.25),
        ]
    )

import pytest

from fal.flags import api_label_index, insert_label, swap_api_label


@pytest.mark.parametrize(
    "host, expected",
    [
        ("api.fal.ai", "rest.fal.ai"),
        ("api.alpha.fal.ai", "rest.alpha.fal.ai"),
        # Environments may add their own labels in front of the service label.
        ("box7.api.staging.example.com", "box7.rest.staging.example.com"),
        # Only the label is matched, never a prefix of one.
        ("apify.example.com", "apify.example.com"),
        ("localhost:8080", "localhost:8080"),
    ],
)
def test_swap_api_label(host, expected):
    assert swap_api_label(host, "rest") == expected


@pytest.mark.parametrize(
    "host, expected",
    [
        ("api.fal.ai", 0),
        ("box7.api.staging.example.com", 1),
        ("localhost:8080", 0),
    ],
)
def test_api_label_index(host, expected):
    assert api_label_index(host) == expected


@pytest.mark.parametrize(
    "run_host, index, expected",
    [
        ("fal.run", 0, "queue.fal.run"),
        ("box7.staging.example.com", 1, "box7.queue.staging.example.com"),
        ("localhost:8080", 0, "queue.localhost:8080"),
    ],
)
def test_insert_label(run_host, index, expected):
    assert insert_label(run_host, "queue", index) == expected

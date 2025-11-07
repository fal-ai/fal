from __future__ import annotations

import os
from base64 import b64encode
from pathlib import Path
from typing import Any, Optional
from unittest.mock import patch

import pytest
from pydantic import BaseModel

from fal.toolkit.file.file import File, GoogleStorageRepository, _try_with_fallback
from fal.toolkit.file.types import FileData, FileRepository


def test_binary_content_matches():
    content = b"Hello World"
    content_base64 = b64encode(content).decode("utf-8")
    file = File.from_bytes(content, repository="in_memory")
    assert file.url.endswith(content_base64)
    assert file.as_bytes() == content


def test_default_content_type():
    file = File.from_bytes(b"Hello World", repository="in_memory")
    assert file.content_type == "application/octet-stream"
    assert file.file_name
    assert file.file_name.endswith(".bin")


def test_file_name_from_content_type():
    file = File.from_bytes(
        b"Hello World", content_type="text/plain", repository="in_memory"
    )
    assert file.content_type == "text/plain"
    assert file.file_name
    assert file.file_name.endswith(".txt")


def test_content_type_from_file_name():
    file = File.from_bytes(
        b"Hello World", file_name="hello.txt", repository="in_memory"
    )
    assert file.content_type == "text/plain"
    assert file.file_name == "hello.txt"


def test_file_size():
    content = b"Hello World"
    file = File.from_bytes(content, repository="in_memory")
    assert file.file_size == len(content)


def test_in_memory_repository_url():
    content = b"Hello World"
    file = File.from_bytes(content, repository="in_memory")
    assert file.url.startswith("data:application/octet-stream;base64,")
    assert file.url.endswith(b64encode(content).decode("utf-8"))


def test_gcp_storage_if_available():
    gcp_sa_json = os.environ.get("GCLOUD_SA_JSON")
    if gcp_sa_json is None:
        pytest.skip(reason="GCLOUD_SA_JSON environment variable is not set")

    gcp_storage = GoogleStorageRepository(
        gcp_account_json=gcp_sa_json, bucket_name="fal_registry_image_results"
    )
    file = File.from_bytes(b"Hello GCP Storage!", repository=gcp_storage)
    assert file.url.startswith(
        "https://storage.googleapis.com/fal_registry_image_results/"
    )


def test_load_nested():
    class Input(BaseModel):
        file: File

    assert (
        Input(file="https://example.com/somefile.txt").file.url
        == "https://example.com/somefile.txt"
    )

    with pytest.raises(ValueError, match="value must be a valid URL"):
        Input(file="not_a_valid_url")

    file_dict = {
        "url": "https://example.com/somefile.txt",
        "content_type": "text/plain",
        "file_name": "somefile.txt",
    }

    parsed_input = Input(file=file_dict)
    assert parsed_input.file.url == file_dict["url"]
    assert parsed_input.file.content_type == file_dict["content_type"]
    assert parsed_input.file.file_name == file_dict["file_name"]


class MockRepository(FileRepository):
    """Mock repository for testing that can be configured to succeed or fail"""

    def __init__(
        self,
        name: str,
        should_fail: bool = False,
        failure_exception: Optional[Exception] = None,
    ):
        self.name = name
        self.should_fail = should_fail
        self.failure_exception = failure_exception or Exception(
            f"Mock failure for {name}"
        )
        self.calls: list[tuple[str, Any, dict[str, Any]]] = []

    def save(self, data: FileData, **kwargs: Any) -> str:
        self.calls.append(("save", data, kwargs))
        if self.should_fail:
            raise self.failure_exception
        return f"success_url_from_{self.name}"

    def save_file(self, file_path: Path, **kwargs: Any) -> tuple[str, FileData]:
        self.calls.append(("save_file", file_path, kwargs))
        if self.should_fail:
            raise self.failure_exception
        return f"success_url_from_{self.name}", FileData(
            b"mock_data", "text/plain", "mock.txt"
        )


class TestTryWithFallback:
    """Test cases for the _try_with_fallback function"""

    def test_success_on_first_attempt(self):
        """Test successful execution on the first repository"""
        mock_repo = MockRepository("primary", should_fail=False)

        with patch(
            "fal.toolkit.file.file.get_builtin_repository", return_value=mock_repo
        ):
            result = _try_with_fallback(
                func="save",
                args=[FileData(b"test_data", "text/plain", "test.txt")],
                repository="primary",
                fallback_repository=None,
                save_kwargs={"key1": "value1"},
                fallback_save_kwargs={},
            )

        assert result == "success_url_from_primary"
        assert len(mock_repo.calls) == 1
        assert mock_repo.calls[0][0] == "save"

    def test_fallback_on_first_failure(self):
        """Test fallback to second repository when first fails"""
        primary_repo = MockRepository("primary", should_fail=True)
        fallback_repo = MockRepository("fallback", should_fail=False)

        with patch("fal.toolkit.file.file.get_builtin_repository") as mock_get_repo:
            mock_get_repo.side_effect = [primary_repo, fallback_repo]

            result = _try_with_fallback(
                func="save",
                args=[FileData(b"test_data", "text/plain", "test.txt")],
                repository="primary",
                fallback_repository="fallback",
                save_kwargs={"key1": "value1"},
                fallback_save_kwargs={"key2": "value2"},
            )

        assert result == "success_url_from_fallback"
        assert len(primary_repo.calls) == 1
        assert len(fallback_repo.calls) == 1
        assert primary_repo.calls[0][2] == {"key1": "value1"}
        assert fallback_repo.calls[0][2] == {"key2": "value2"}

    def test_fallback_with_list_of_repositories(self):
        """Test fallback through a list of repositories"""
        repo1 = MockRepository("repo1", should_fail=True)
        repo2 = MockRepository("repo2", should_fail=True)
        repo3 = MockRepository("repo3", should_fail=False)

        with patch("fal.toolkit.file.file.get_builtin_repository") as mock_get_repo:
            mock_get_repo.side_effect = [repo1, repo2, repo3]

            result = _try_with_fallback(
                func="save",
                args=[FileData(b"test_data", "text/plain", "test.txt")],
                repository="repo1",
                fallback_repository=["repo2", "repo3"],
                save_kwargs={"key1": "value1"},
                fallback_save_kwargs={"key2": "value2"},
            )

        assert result == "success_url_from_repo3"
        assert len(repo1.calls) == 1
        assert len(repo2.calls) == 1
        assert len(repo3.calls) == 1

    def test_all_repositories_fail(self):
        """Test that exception is raised when all repositories fail"""
        repo1 = MockRepository("repo1", should_fail=True)
        repo2 = MockRepository("repo2", should_fail=True)

        with patch("fal.toolkit.file.file.get_builtin_repository") as mock_get_repo:
            mock_get_repo.side_effect = [repo1, repo2]

            with pytest.raises(Exception, match="Mock failure for repo2"):
                _try_with_fallback(
                    func="save",
                    args=[FileData(b"test_data", "text/plain", "test.txt")],
                    repository="repo1",
                    fallback_repository="repo2",
                    save_kwargs={},
                    fallback_save_kwargs={},
                )

    def test_no_fallback_repository(self):
        """Test behavior when no fallback repository is provided"""
        repo = MockRepository("primary", should_fail=True)

        with patch("fal.toolkit.file.file.get_builtin_repository", return_value=repo):
            with pytest.raises(Exception, match="Mock failure for primary"):
                _try_with_fallback(
                    func="save",
                    args=[FileData(b"test_data", "text/plain", "test.txt")],
                    repository="primary",
                    fallback_repository=None,
                    save_kwargs={},
                    fallback_save_kwargs={},
                )

    def test_save_file_function(self):
        """Test with save_file function instead of save"""
        mock_repo = MockRepository("primary", should_fail=False)
        test_path = Path("/tmp/test.txt")

        with patch(
            "fal.toolkit.file.file.get_builtin_repository", return_value=mock_repo
        ):
            result = _try_with_fallback(
                func="save_file",
                args=[test_path],
                repository="primary",
                fallback_repository=None,
                save_kwargs={"content_type": "text/plain"},
                fallback_save_kwargs={},
            )

        assert result[0] == "success_url_from_primary"
        assert isinstance(result[1], FileData)
        assert result[1].data == b"mock_data"
        assert result[1].content_type == "text/plain"
        assert result[1].file_name == "mock.txt"
        assert len(mock_repo.calls) == 1
        assert mock_repo.calls[0][0] == "save_file"
        assert mock_repo.calls[0][1] == test_path

    def test_custom_exception_types(self):
        """Test with different types of exceptions"""
        custom_exception = ValueError("Custom error message")
        repo1 = MockRepository(
            "repo1", should_fail=True, failure_exception=custom_exception
        )
        repo2 = MockRepository("repo2", should_fail=False)

        with patch("fal.toolkit.file.file.get_builtin_repository") as mock_get_repo:
            mock_get_repo.side_effect = [repo1, repo2]

            result = _try_with_fallback(
                func="save",
                args=[FileData(b"test_data", "text/plain", "test.txt")],
                repository="repo1",
                fallback_repository="repo2",
                save_kwargs={},
                fallback_save_kwargs={},
            )

        assert result == "success_url_from_repo2"

    def test_empty_fallback_list(self):
        """Test with empty fallback list"""
        repo = MockRepository("primary", should_fail=True)

        with patch("fal.toolkit.file.file.get_builtin_repository", return_value=repo):
            with pytest.raises(Exception, match="Mock failure for primary"):
                _try_with_fallback(
                    func="save",
                    args=[FileData(b"test_data", "text/plain", "test.txt")],
                    repository="primary",
                    fallback_repository=[],
                    save_kwargs={},
                    fallback_save_kwargs={},
                )

    def test_repository_id_vs_object(self):
        """Test that both repository IDs and repository objects work"""
        mock_repo = MockRepository("test_repo", should_fail=False)

        # Test with repository ID
        with patch(
            "fal.toolkit.file.file.get_builtin_repository", return_value=mock_repo
        ):
            result1 = _try_with_fallback(
                func="save",
                args=[FileData(b"test_data", "text/plain", "test.txt")],
                repository="test_repo",
                fallback_repository=None,
                save_kwargs={},
                fallback_save_kwargs={},
            )

        # Test with repository object
        result2 = _try_with_fallback(
            func="save",
            args=[FileData(b"test_data", "text/plain", "test.txt")],
            repository=mock_repo,
            fallback_repository=None,
            save_kwargs={},
            fallback_save_kwargs={},
        )

        assert result1 == "success_url_from_test_repo"
        assert result2 == "success_url_from_test_repo"

    def test_traceback_and_print_output(self):
        """Test that traceback and print statements are called on failure"""
        repo1 = MockRepository("repo1", should_fail=True)
        repo2 = MockRepository("repo2", should_fail=False)

        with patch(
            "fal.toolkit.file.file.get_builtin_repository"
        ) as mock_get_repo, patch(
            "fal.toolkit.file.file.traceback.print_exc"
        ) as mock_traceback, patch("builtins.print") as mock_print:
            mock_get_repo.side_effect = [repo1, repo2]

            result = _try_with_fallback(
                func="save",
                args=[FileData(b"test_data", "text/plain", "test.txt")],
                repository="repo1",
                fallback_repository="repo2",
                save_kwargs={},
                fallback_save_kwargs={},
            )

        assert result == "success_url_from_repo2"
        mock_traceback.assert_called_once()
        mock_print.assert_called_once()
        # Check that the print message contains the expected text
        print_call_args = mock_print.call_args[0][0]
        assert "Failed to save to repository repo1" in print_call_args
        assert "falling back to repo2" in print_call_args

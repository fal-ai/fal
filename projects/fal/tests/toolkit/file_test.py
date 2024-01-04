from __future__ import annotations

import os
from base64 import b64encode
from pathlib import Path

import pytest
from fal.toolkit.file.file import File, GoogleStorageRepository
from fal.toolkit import CompressedFile
from pydantic import BaseModel


def test_binary_content_matches():
    content = b"Hello World"
    content_base64 = b64encode(content).decode("utf-8")
    file = File.from_bytes(content, repository="in_memory")
    assert file.url.endswith(content_base64)
    assert file.as_bytes() == content


def test_default_content_type():
    file = File.from_bytes(b"Hello World", repository="in_memory")
    assert file.content_type == "application/octet-stream"
    assert file.file_name.endswith(".bin")


def test_file_name_from_content_type():
    file = File.from_bytes(
        b"Hello World", content_type="text/plain", repository="in_memory"
    )
    assert file.content_type == "text/plain"
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


def test_compressed_file():
    class TestInput(BaseModel):
        files: CompressedFile

    archive_url = "https://storage.googleapis.com/falserverless/sdk_tests/compressed_file_test.zip"

    test_input = TestInput(files=archive_url)

    extracted_file_paths = [file for file in test_input.files]

    assert all(isinstance(file, Path) for file in extracted_file_paths)
    assert len(extracted_file_paths) == 3

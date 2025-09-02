import json
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, mock_open, patch

import pytest

from fal.file_sync import FileSync


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def file_sync(temp_dir):
    """Create FileSync instance with temporary directory"""
    return FileSync(local_file_path=str(Path(temp_dir) / "app.py"))


@pytest.fixture
def mock_credentials():
    """Mock credentials for testing"""
    mock_creds = MagicMock()
    mock_creds.to_headers.return_value = {"Authorization": "Bearer test-token"}
    return mock_creds


def test_file_sync_init(temp_dir):
    """Test FileSync initialization sets correct paths"""
    local_path = str(Path(temp_dir) / "test_app.py")
    fs = FileSync(local_path)

    assert (
        fs.local_file_path == local_path
    ), "Incorrect local_file_path, base file is not set correctly"


@patch("pathlib.Path.mkdir")
def test_file_sync_init_creates_cache_dir(mock_mkdir, temp_dir):
    """Test that cache directory is created on initialization"""
    local_path = str(Path(temp_dir) / "test_app.py")
    fs = FileSync(local_path)

    # Access cache_dir to trigger mkdir
    _ = fs.cache_dir
    (
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True),
        "Cache directory should be created with parents=True and exist_ok=True",
    )


@pytest.mark.asyncio
async def test_request_success(file_sync):
    """Test successful HTTP request returns response"""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"success": True}

    with patch.object(file_sync, "_client") as mock_client:
        mock_client.request = AsyncMock(return_value=mock_response)

        response = await file_sync._request("GET", "/test", param="value")

        assert response == mock_response, "Should return the mock response object"
        (
            mock_client.request.assert_called_once_with("GET", "/test", param="value"),
            "Should call client.request with correct parameters",
        )


@pytest.mark.asyncio
async def test_request_failure_with_json_detail(file_sync):
    """Test HTTP request failure raises exception with JSON error detail"""
    from fal.exceptions import FalServerlessException

    mock_response = MagicMock()
    mock_response.status_code = 400
    mock_response.json.return_value = {"detail": "Bad request"}

    with patch.object(file_sync, "_client") as mock_client:
        mock_client.request = AsyncMock(return_value=mock_response)

        with pytest.raises(FalServerlessException, match="Bad request") as exc_info:
            await file_sync._request("GET", "/test")

        assert "Bad request" in str(
            exc_info.value
        ), "Exception should contain the JSON error detail"


@pytest.mark.asyncio
async def test_request_failure_with_text_detail(file_sync):
    """Test HTTP request failure falls back to text when JSON parsing fails"""
    from fal.exceptions import FalServerlessException

    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.json.side_effect = Exception("JSON decode error")
    mock_response.text = "Internal server error"

    with patch.object(file_sync, "_client") as mock_client:
        mock_client.request = AsyncMock(return_value=mock_response)

        with pytest.raises(
            FalServerlessException, match="Internal server error"
        ) as exc_info:
            await file_sync._request("GET", "/test")

        assert "Internal server error" in str(
            exc_info.value
        ), "Exception should fall back to text content when JSON parsing fails"


# HASH COMPUTATION TESTS


def test_compute_hash_basic_functionality(file_sync, temp_dir):
    """Test hash computation produces valid SHA256"""
    test_file = Path(temp_dir) / "test.txt"
    test_content = "Hello, world!"
    test_file.write_text(test_content)

    stat = os.stat(test_file)
    hash_result = file_sync.compute_hash(str(test_file), stat.st_mode)

    assert len(hash_result) == 64, "Expected SHA256 hash length of 64 chars"
    assert all(
        c in "0123456789abcdef" for c in hash_result
    ), "Hash should only contain hex chars"


def test_compute_hash_metadata_affects_result(file_sync, temp_dir):
    """Test that file metadata changes affect hash computation"""
    test_file = Path(temp_dir) / "test.txt"
    test_file.write_text("same content")

    stat1 = os.stat(test_file)
    hash1 = file_sync.compute_hash(str(test_file), stat1.st_mode)
    hash2 = file_sync.compute_hash(str(test_file), stat1.st_mode + 1)

    assert hash1 != hash2, "Different metadata should produce different hashes"


def test_compute_hash_large_file_chunking(file_sync, temp_dir):
    """Test hash computation handles large files (tests chunking logic)"""
    test_file = Path(temp_dir) / "large_file.txt"
    # Create file larger than chunk size (4096 bytes)
    large_content = "x" * 10000
    test_file.write_text(large_content)

    stat = os.stat(test_file)
    hash_result = file_sync.compute_hash(str(test_file), stat.st_mode)

    assert len(hash_result) == 64, "Large file hash should still be 64 chars"
    # Verify it's different from a smaller file with same pattern
    small_file = Path(temp_dir) / "small_file.txt"
    small_file.write_text("x" * 100)
    small_stat = os.stat(small_file)
    small_hash = file_sync.compute_hash(str(small_file), small_stat.st_mode)
    assert (
        hash_result != small_hash
    ), "Large and small files with different content should have different hashes"


# FILE METADATA TESTS


def test_get_file_metadata_structure(file_sync, temp_dir):
    """Test file metadata returns correct structure and values"""
    test_file = Path(temp_dir) / "test.txt"
    test_content = "test content"
    test_file.write_text(test_content)

    metadata = file_sync.get_file_metadata(str(test_file))

    required_keys = ["size", "mtime", "mode", "hash"]
    for key in required_keys:
        assert (
            key in metadata
        ), f"Metadata missing required key: {key}. Got keys: {list(metadata.keys())}"

    assert int(metadata["size"]) == len(test_content), "Incorrect file size"
    assert len(metadata["hash"]) == 64, "Incorrect hash length"


def test_normalize_path_relative_to_base(file_sync, temp_dir):
    """Test path normalization resolves relative paths correctly"""
    base_file = str(Path(temp_dir) / "app.py")
    fs = FileSync(base_file)

    abs_path, rel_path = fs.normalize_path("config.txt", base_file)

    expected_abs = (Path(temp_dir) / "config.txt").resolve().as_posix()
    assert abs_path == expected_abs, "Incorrect absolute path when normalizing file"
    assert rel_path == "config.txt", "Expected relative path 'config.txt'"


def test_normalize_path_absolute_unchanged(file_sync, temp_dir):
    """Test that absolute paths remain unchanged"""
    absolute_path = "/absolute/path/file.txt"

    abs_path, rel_path = file_sync.normalize_path(absolute_path, temp_dir)

    assert abs_path == absolute_path, "Absolute path should remain unchanged"


def test_normalize_path_security_parent_dir_removal(file_sync, temp_dir):
    """Test that parent directory references are sanitized for security"""
    abs_path, rel_path = file_sync.normalize_path("../secret.txt", temp_dir)

    assert (
        "../" not in rel_path
    ), "Parent directory references should be removed from relative path"
    assert "secret.txt" in rel_path, "Filename should still be present in relative path"


# CACHE MANAGEMENT TESTS


@pytest.mark.asyncio
async def test_fetch_cache_from_server_success(file_sync):
    """Test successful cache retrieval from server"""
    expected_cache = [
        {"hash": "abc123", "mtime": "1234567890", "mode": "33188", "size_bytes": "100"}
    ]

    with patch.object(
        file_sync,
        "_request",
        AsyncMock(return_value=MagicMock(json=lambda: expected_cache)),
    ):
        cache = await file_sync.fetch_cache_from_server()

        assert cache == expected_cache, "Should return server cache data."


@pytest.mark.asyncio
async def test_fetch_cache_from_server_error_returns_empty(file_sync):
    """Test error handling when server cache fetch fails"""
    with patch.object(
        file_sync, "_request", AsyncMock(side_effect=Exception("Network error"))
    ):
        cache = await file_sync.fetch_cache_from_server()

        assert cache == [], "Should return empty list when server fetch fails"


def test_save_local_cache_creates_valid_json(file_sync, temp_dir):
    """Test local cache save creates valid JSON file"""
    cache_data = {"hash123": {"size": "100", "mtime": "123456"}}
    cache_file = Path(temp_dir) / "cache.json"

    with patch.object(file_sync, "cache_file", cache_file):
        file_sync.save_local_cache(cache_data)

        assert cache_file.exists(), "Cache file should exist"

        with open(cache_file) as f:
            saved_data = json.load(f)

        assert saved_data == cache_data, "Saved data should match input"


@pytest.mark.asyncio
async def test_load_local_cache_existing_file(file_sync, temp_dir):
    """Test loading cache from existing local file"""
    cache_data = {"hash123": {"size": "100", "mtime": "123456"}}
    cache_file = Path(temp_dir) / "cache.json"

    # Create cache file
    with open(cache_file, "w") as f:
        json.dump(cache_data, f)

    with patch.object(file_sync, "cache_file", cache_file):
        loaded_cache = await file_sync.load_local_cache()

        assert loaded_cache == cache_data, "Loaded cache should match saved data."


@pytest.mark.asyncio
async def test_load_local_cache_fallback_to_server(file_sync, temp_dir):
    """Test cache loading falls back to server when local cache missing"""
    server_cache = [
        {"hash": "abc123", "mtime": "1234567890", "mode": "33188", "size_bytes": "100"}
    ]
    expected_transformed = {
        "abc123": {
            "mtime": "1234567890",
            "mode": "33188",
            "size": "100",
            "hash": "abc123",
        }
    }

    with patch.object(
        file_sync, "cache_file", Path(temp_dir) / "nonexistent.json"
    ), patch.object(
        file_sync, "fetch_cache_from_server", AsyncMock(return_value=server_cache)
    ), patch.object(file_sync, "save_local_cache") as mock_save:
        loaded_cache = await file_sync.load_local_cache()

        assert (
            loaded_cache == expected_transformed
        ), "Should transform server cache format."
        (
            mock_save.assert_called_once_with(expected_transformed),
            "Should save transformed cache locally",
        )


@pytest.mark.asyncio
async def test_check_local_cache_hit_vs_miss(file_sync):
    """Test cache hit detection logic"""
    current_metadata = {"hash": "abc123", "mtime": "123", "size": "100"}

    # Test cache hit
    hit_cache = {"abc123": {"hash": "abc123", "mtime": "123", "size": "100"}}
    with patch.object(file_sync, "load_local_cache", AsyncMock(return_value=hit_cache)):
        result = await file_sync.check_local_cache(current_metadata)
        assert result is True, "Should return True for cache hit with matching metadata"

    # Test cache miss - hash not found
    miss_cache = {"different_hash": {"hash": "different_hash"}}
    with patch.object(
        file_sync, "load_local_cache", AsyncMock(return_value=miss_cache)
    ):
        result = await file_sync.check_local_cache(current_metadata)
        assert result is False, "Should return False when hash not found in cache"

    # Test cache miss - hash differs
    corrupt_cache = {"abc123": {"hash": "different_hash"}}
    with patch.object(
        file_sync, "load_local_cache", AsyncMock(return_value=corrupt_cache)
    ):
        result = await file_sync.check_local_cache(current_metadata)
        assert (
            result is False
        ), "Should return False when cached hash differs from current"


@pytest.mark.asyncio
async def test_update_local_cache_adds_timestamp(file_sync):
    """Test cache update adds new entry with timestamp"""
    metadata = {"hash": "abc123", "size": "100", "mtime": "123"}
    existing_cache = {"old_hash": {"hash": "old_hash"}}

    with patch.object(
        file_sync, "load_local_cache", AsyncMock(return_value=existing_cache)
    ), patch.object(file_sync, "save_local_cache") as mock_save, patch(
        "asyncio.get_event_loop"
    ) as mock_loop:
        mock_loop.return_value.time.return_value = 1234567890.0

        await file_sync.update_local_cache(metadata)

        expected_cache = {
            "old_hash": {"hash": "old_hash"},
            "abc123": {
                "hash": "abc123",
                "size": "100",
                "mtime": "123",
                "cached_at": "1234567890.0",
            },
        }

        (
            mock_save.assert_called_once_with(expected_cache),
            f"Should save cache with timestamp. Expected {expected_cache}",
        )


# FILE COLLECTION TESTS


def test_collect_files_single_file_and_directory(file_sync, temp_dir):
    """Test collecting both individual files and directories"""
    # Create test structure
    app_file = Path(temp_dir) / "app.py"
    app_file.write_text("# app file")

    single_file = Path(temp_dir) / "readme.txt"
    single_file.write_text("readme content")

    config_dir = Path(temp_dir) / "config"
    config_dir.mkdir()
    (config_dir / "config.json").write_text('{"key": "value"}')

    fs = FileSync(str(app_file))
    files = fs.collect_files(["readme.txt", "config/"])

    assert len(files) == 2, "Expected 2 files (1 single + 1 from directory)"

    relative_paths = [f["relative_path"] for f in files]
    assert "readme.txt" in relative_paths, "Missing readme.txt in collected files"
    assert (
        "config/config.json" in relative_paths
    ), "Missing config/config.json in collected files"

    # Verify metadata structure
    for file_info in files:
        required_keys = ["absolute_path", "relative_path", "hash", "size"]
        for key in required_keys:
            assert (
                key in file_info
            ), f"File info missing required key '{key}': {file_info}"


def test_collect_files_handles_nonexistent_gracefully(file_sync, temp_dir):
    """Test that nonexistent files/directories are skipped without error"""
    app_file = Path(temp_dir) / "app.py"
    app_file.write_text("# app file")

    fs = FileSync(str(app_file))
    files = fs.collect_files(["nonexistent.txt", "missing_dir/"])

    assert len(files) == 0, "Should skip nonexistent paths without error"


# HASH EXISTENCE TESTS


@pytest.mark.asyncio
async def test_check_hash_exists_status_codes(file_sync):
    """Test hash existence check handles different HTTP status codes"""
    test_cases = [
        (200, True, "Should return True for 200 OK"),
        (404, False, "Should return False for 404 Not Found"),
    ]

    for status_code, expected, message in test_cases:
        mock_response = MagicMock()
        mock_response.status_code = status_code

        with patch.object(file_sync, "_request", AsyncMock(return_value=mock_response)):
            result = await file_sync.check_hash_exists("test_hash", "test_file")
            assert (
                result is expected
            ), f"{message}. Status {status_code} should return {expected}"


@pytest.mark.asyncio
async def test_check_hash_exists_error_handling(file_sync):
    """Test hash existence check handles network errors gracefully"""
    with patch.object(
        file_sync, "_request", AsyncMock(side_effect=Exception("Network error"))
    ):
        result = await file_sync.check_hash_exists("abc123", "abc123")
        assert (
            result is False
        ), "Should return False on network errors, not raise exception"


@pytest.mark.asyncio
async def test_check_multiple_hashes_exist_bulk_operation(file_sync):
    """Test bulk hash existence checking with mixed results"""

    def mock_check_side_effect(hash_val, file_name):
        return hash_val in ["existing_hash1", "existing_hash3"]

    with patch.object(
        file_sync, "check_hash_exists", side_effect=mock_check_side_effect
    ):
        result = await file_sync.check_multiple_hashes_exist(
            ["existing_hash1", "missing_hash2", "existing_hash3"],
            ["existing_file1", "missing_file2", "exsiting_file3"],
        )

        expected = {
            "existing_hash1": True,
            "missing_hash2": False,
            "existing_hash3": True,
        }
        assert (
            result == expected
        ), "Bulk hash check should return correct status for each hash."


# FILE UPLOAD TESTS


def test_upload_file_tus_basic_functionality(file_sync, temp_dir):
    """Test TUS file upload calls correct methods with parameters"""
    test_file = Path(temp_dir) / "test.txt"
    test_file.write_text("test content")

    mock_uploader = MagicMock()
    expected_url = "https://upload.server.com/files/abc123"
    mock_uploader.url = expected_url

    with patch.object(file_sync, "_tus_client") as mock_client:
        mock_client.uploader.return_value = mock_uploader

        result_url = file_sync.upload_file_tus(
            str(test_file), chunk_size=1024, metadata={"test": "value"}
        )

        assert result_url == expected_url, "Should return uploader URL"
        (
            mock_client.uploader.assert_called_once_with(
                str(test_file), chunk_size=1024, metadata={"test": "value"}
            ),
            "Should call uploader with correct parameters",
        )
        (
            mock_uploader.upload.assert_called_once(),
            "Should call upload on the uploader instance",
        )


@pytest.mark.asyncio
async def test_upload_multiple_files_success_and_error_handling(file_sync, temp_dir):
    """Test multiple file upload handles both success and failure cases"""
    file1 = Path(temp_dir) / "good_file.txt"
    file2 = Path(temp_dir) / "bad_file.txt"
    file1.write_text("content1")
    file2.write_text("content2")

    files_metadata = [
        {
            "absolute_path": str(file1),
            "relative_path": "good_file.txt",
            "hash": "hash1",
        },
        {"absolute_path": str(file2), "relative_path": "bad_file.txt", "hash": "hash2"},
    ]

    def mock_upload_side_effect(path, **kwargs):
        if "bad_file" in path:
            raise Exception("Upload failed")
        return f"https://upload.server.com/files/{Path(path).name}"

    with patch.object(
        file_sync, "upload_file_tus", side_effect=mock_upload_side_effect
    ):
        results = await file_sync.upload_multiple_files(files_metadata)

        assert len(results) == 2, "Should return results for both files"

        # Check successful upload
        success_result = next(
            r for r in results if r["relative_path"] == "good_file.txt"
        )
        assert (
            "upload_url" in success_result
        ), "Successful upload should have upload_url"

        # Check failed upload
        error_result = next(r for r in results if r["relative_path"] == "bad_file.txt")
        assert "error" in error_result, "Failed upload should have error field"
        assert (
            "Upload failed" in error_result["error"]
        ), "Error message should describe the failure"


# INTEGRATION TESTS


@pytest.mark.asyncio
async def test_sync_files_complete_workflow(file_sync, temp_dir):
    """Test end-to-end file sync workflow with mixed scenarios"""
    # Setup test files
    file1 = Path(temp_dir) / "cached_file.txt"
    file2 = Path(temp_dir) / "server_file.txt"
    file3 = Path(temp_dir) / "upload_file.txt"

    for i, f in enumerate([file1, file2, file3], 1):
        f.write_text(f"content{i}")

    fs = FileSync(str(Path(temp_dir) / "app.py"))

    files_metadata = [
        {
            "absolute_path": str(file1),
            "relative_path": "cached_file.txt",
            "hash": "cached_hash",
        },
        {
            "absolute_path": str(file2),
            "relative_path": "server_file.txt",
            "hash": "server_hash",
        },
        {
            "absolute_path": str(file3),
            "relative_path": "upload_file.txt",
            "hash": "upload_hash",
        },
    ]

    # Mock different cache states
    def mock_cache_check(metadata):
        return metadata["hash"] == "cached_hash"

    with patch.object(fs, "collect_files", return_value=files_metadata), patch.object(
        fs, "check_local_cache", side_effect=mock_cache_check
    ), patch.object(
        fs,
        "check_multiple_hashes_exist",
        AsyncMock(return_value={"server_hash": True, "upload_hash": False}),
    ), patch.object(
        fs,
        "upload_multiple_files",
        AsyncMock(
            return_value=[{"upload_url": "http://uploaded", "hash": "upload_hash"}]
        ),
    ), patch.object(fs, "update_local_cache", AsyncMock()) as mock_update:
        results = await fs.sync_files(
            ["cached_file.txt", "server_file.txt", "upload_file.txt"]
        )

        assert (
            len(results["existing_hashes"]) == 2
        ), "Should have 2 existing hashes (cached + server)"
        assert len(results["uploaded_files"]) == 1, "Should have 1 uploaded file"
        assert len(results["errors"]) == 0, "Should have no errors"

        # Verify cache updates for server hash
        assert (
            mock_update.call_count >= 1
        ), "Should update cache for files found on server"


@pytest.mark.asyncio
async def test_sync_files_deduplication_logic(file_sync, temp_dir):
    """Test that duplicate files in input are properly deduplicated"""
    file1 = Path(temp_dir) / "duplicate_file.txt"
    file1.write_text("content")

    fs = FileSync(str(Path(temp_dir) / "app.py"))

    # Mock collect_files to return duplicates (same hash)
    duplicate_metadata = [
        {
            "absolute_path": str(file1),
            "relative_path": "duplicate_file.txt",
            "hash": "same_hash",
        },
        {
            "absolute_path": str(file1),
            "relative_path": "duplicate_file.txt",
            "hash": "same_hash",
        },
    ]

    with patch.object(
        fs, "collect_files", return_value=duplicate_metadata
    ), patch.object(fs, "check_local_cache", AsyncMock(return_value=True)):
        results = await fs.sync_files(["duplicate_file.txt", "duplicate_file.txt"])

        assert (
            len(results["existing_hashes"]) == 1
        ), "Duplicates should be deduplicated to 1 entry"


# SIMPLIFIED ERROR HANDLING TESTS


@pytest.mark.asyncio
async def test_close_client(file_sync):
    """Test HTTP client cleanup"""
    mock_client = AsyncMock()

    with patch.object(file_sync, "_client", mock_client):
        await file_sync.close()
        mock_client.aclose.assert_called_once(), "Should close the async HTTP client"


def test_save_local_cache_handles_io_errors_gracefully(file_sync):
    """Test cache save handles IO errors without crashing"""
    cache_data = {"hash123": {"size": "100"}}

    with patch("builtins.open", mock_open()) as mock_file:
        mock_file.side_effect = OSError("Permission denied")

        # Should not raise exception
        try:
            file_sync.save_local_cache(cache_data)
        except Exception as e:
            pytest.fail(
                f"save_local_cache should handle IO errors gracefully, but raised: {e}"
            )

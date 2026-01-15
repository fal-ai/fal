import re
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import fal.file_sync as file_sync_mod
from fal.file_sync import FileSync


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def file_sync(temp_dir, monkeypatch):
    """Create FileSync instance with temporary directory"""
    # Ensure credential and host resolution do not require real auth or HTTPS
    monkeypatch.setenv("FAL_KEY", "dummy:dummy")
    monkeypatch.setenv("FAL_HOST", "api.localhost")
    monkeypatch.setenv("ISOLATE_TEST_MODE", "1")
    return FileSync(local_file_path=str(Path(temp_dir) / "app.py"))


def test_file_sync_init(temp_dir):
    """Test FileSync initialization sets correct paths"""
    local_path = str(Path(temp_dir) / "test_app.py")
    fs = FileSync(local_path)
    assert fs.local_file_path == local_path


def test_request_success(file_sync):
    """Test successful HTTP request returns response"""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"success": True}

    with patch.object(file_sync, "_client") as mock_client:
        mock_client.request.return_value = mock_response
        response = file_sync._request("GET", "/test")
        assert response == mock_response


def test_request_failure_with_json_detail(file_sync):
    """Test HTTP request failure raises exception with JSON error detail"""
    from fal.exceptions import FalServerlessException

    mock_response = MagicMock()
    mock_response.status_code = 400
    mock_response.json.return_value = {"detail": "Bad request"}

    with patch.object(file_sync, "_client") as mock_client:
        mock_client.request.return_value = mock_response
        with pytest.raises(FalServerlessException, match="Bad request"):
            file_sync._request("GET", "/test")


def test_request_404_raises_not_found(file_sync):
    """Test HTTP 404 raises specific not found exception"""
    from fal.exceptions import FalServerlessException

    mock_response = MagicMock()
    mock_response.status_code = 404

    with patch.object(file_sync, "_client") as mock_client:
        mock_client.request.return_value = mock_response
        with pytest.raises(FalServerlessException, match="Not Found"):
            file_sync._request("GET", "/test")


def test_compute_hash_basic_functionality(temp_dir):
    """Test hash computation produces valid SHA256"""
    test_file = Path(temp_dir) / "test.txt"
    test_file.write_text("Hello, world!")

    stat = test_file.stat()
    hash_result = file_sync_mod.compute_hash(test_file, stat.st_mode)

    assert len(hash_result) == 64
    assert all(c in "0123456789abcdef" for c in hash_result)


def test_compute_hash_different_content_different_hash(temp_dir):
    """Test different file content produces different hashes"""
    file1 = Path(temp_dir) / "file1.txt"
    file2 = Path(temp_dir) / "file2.txt"
    file1.write_text("content1")
    file2.write_text("content2")

    hash1 = file_sync_mod.compute_hash(file1, file1.stat().st_mode)
    hash2 = file_sync_mod.compute_hash(file2, file2.stat().st_mode)

    assert hash1 != hash2


def test_get_file_metadata_structure(temp_dir):
    """Test file metadata returns correct structure and values"""
    test_file = Path(temp_dir) / "test.txt"
    test_content = "test content"
    test_file.write_text(test_content)

    metadata = file_sync_mod.FileMetadata.from_path(
        test_file, relative="test.txt", absolute=str(test_file)
    )

    assert metadata.size == len(test_content)
    assert len(metadata.hash) == 64


def test_get_file_metadata_file_too_large_error(temp_dir):
    """Test that files exceeding size limit raise FileTooLargeError"""
    from fal.exceptions import FileTooLargeError

    test_file = Path(temp_dir) / "test.txt"
    test_file.write_text("content")

    # Mock the stat to return a size larger than FILE_SIZE_LIMIT
    with patch("pathlib.Path.stat") as mock_stat:
        mock_stat.return_value.st_size = 1024 * 1024 * 1024 + 1  # 1GB + 1 byte
        mock_stat.return_value.st_mtime = 123456789
        mock_stat.return_value.st_mode = 33188

        with pytest.raises(FileTooLargeError):
            file_sync_mod.FileMetadata.from_path(
                test_file, relative="test.txt", absolute=str(test_file)
            )


def test_normalize_path_relative_to_base(temp_dir):
    """Test path normalization resolves relative paths correctly"""
    base_file = str(Path(temp_dir) / "app.py")

    abs_path, rel_path = file_sync_mod.normalize_path("config.txt", base_file)

    expected_abs = (Path(temp_dir) / "config.txt").resolve().as_posix()
    assert abs_path == expected_abs
    assert rel_path == "config.txt"


def test_collect_files_single_file(temp_dir):
    """Test collecting a single file"""
    app_file = Path(temp_dir) / "app.py"
    app_file.write_text("# app file")

    single_file = Path(temp_dir) / "readme.txt"
    single_file.write_text("readme content")

    fs = FileSync(str(app_file))
    files = fs.collect_files(["readme.txt"])

    assert len(files) == 1
    assert files[0].relative_path == "readme.txt"


def test_collect_files_directory(temp_dir):
    """Test collecting files from a directory"""
    app_file = Path(temp_dir) / "app.py"
    app_file.write_text("# app file")

    config_dir = Path(temp_dir) / "config"
    config_dir.mkdir()
    (config_dir / "config.json").write_text('{"key": "value"}')
    (config_dir / "settings.yml").write_text("key: value")

    fs = FileSync(str(app_file))
    files = fs.collect_files(["config/"])

    assert len(files) == 2
    relative_paths = [f.relative_path for f in files]
    assert "config/config.json" in relative_paths
    assert "config/settings.yml" in relative_paths


def test_collect_files_handles_nonexistent_gracefully(temp_dir):
    """Test that nonexistent files/directories are skipped without error"""
    app_file = Path(temp_dir) / "app.py"
    app_file.write_text("# app file")

    fs = FileSync(str(app_file))
    files = fs.collect_files(["nonexistent.txt", "missing_dir/"])

    assert len(files) == 0


def test_should_ignore_file_basic_patterns(file_sync):
    """Test basic regex patterns for ignoring files"""
    ignore_patterns_str = [r"\.pyc$", r"__pycache__/", r"\.git/"]
    ignore_patterns = [re.compile(pattern) for pattern in ignore_patterns_str]

    # Should ignore
    assert file_sync._matches_patterns("test.pyc", ignore_patterns)
    assert file_sync._matches_patterns("__pycache__/file.py", ignore_patterns)
    assert file_sync._matches_patterns(".git/config", ignore_patterns)

    # Should not ignore
    assert not file_sync._matches_patterns("test.py", ignore_patterns)
    assert not file_sync._matches_patterns("src/main.py", ignore_patterns)


def test_should_ignore_file_empty_patterns(file_sync):
    """Test behavior with empty ignore patterns"""
    result = file_sync._matches_patterns("test.py", [])
    assert not result


def test_check_hashes_on_server_success(file_sync):
    """Test successful hash check on server"""
    test_hashes = ["hash1", "hash2", "hash3"]
    missing_hashes = ["hash2"]

    mock_response = MagicMock()
    mock_response.json.return_value = missing_hashes

    with patch.object(file_sync, "_request", return_value=mock_response):
        result = file_sync.check_hashes_on_server(test_hashes)
        assert result == missing_hashes


def test_check_hashes_on_server_error_returns_all_missing(file_sync):
    """Test hash check handles server errors gracefully"""
    test_hashes = ["hash1", "hash2"]

    with patch.object(file_sync, "_request", side_effect=Exception("Network error")):
        result = file_sync.check_hashes_on_server(test_hashes)
        assert result == test_hashes


def test_upload_file_multipart_basic_functionality(file_sync, temp_dir):
    """Test multipart file upload calls correct methods with parameters"""
    test_file = Path(temp_dir) / "test.txt"
    test_file.write_text("test content")

    metadata = file_sync_mod.FileMetadata.from_path(
        test_file, relative="test.txt", absolute=str(test_file)
    )

    with patch("fal.file_sync.AppFileMultipartUpload") as MockMultipart:
        mock_instance = MagicMock()
        mock_instance.upload_file.return_value = "test_etag"
        MockMultipart.return_value = mock_instance

        result = file_sync.upload_file_multipart(str(test_file), metadata)

        assert result == "test_etag"
        MockMultipart.assert_called_once()
        mock_instance.upload_file.assert_called_once_with(str(test_file))


def test_multipart_upload_returns_md5_etag(temp_dir):
    """Test that multipart upload returns MD5 etag from server"""
    from fal.upload import AppFileMultipartUpload

    test_file = Path(temp_dir) / "test.txt"
    test_file.write_text("test content")

    metadata = file_sync_mod.FileMetadata.from_path(
        test_file, relative="test.txt", absolute=str(test_file)
    )

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 200

    # Simulate server responses
    initiate_response = MagicMock()
    initiate_response.status_code = 200
    initiate_response.json.return_value = {"upload_id": "test-upload-id"}

    part_response = MagicMock()
    part_response.status_code = 200
    part_response.json.return_value = {
        "part_number": 1,
        "etag": "d8e8fca2dc0f896fd7cb4cb0031ba249",  # MD5 hash (32 chars)
    }

    complete_response = MagicMock()
    complete_response.status_code = 200
    complete_response.json.return_value = {
        "etag": "d8e8fca2dc0f896fd7cb4cb0031ba249"  # MD5, not SHA256
    }

    mock_client.request.side_effect = [
        initiate_response,
        part_response,
        complete_response,
    ]

    uploader = AppFileMultipartUpload(
        client=mock_client,
        file_hash=metadata.hash,  # SHA256 (64 chars)
        metadata=metadata.to_dict(),
    )

    etag = uploader.upload_file(str(test_file))

    # Server returns MD5 (32 chars), not SHA256 (64 chars)
    assert len(etag) == 32
    assert etag == "d8e8fca2dc0f896fd7cb4cb0031ba249"
    # No client-side hash verification should happen
    # (would fail if we compared MD5 to SHA256)


def test_multipart_upload_bounded_queue_limits_memory(temp_dir):
    """Test that bounded queue prevents loading entire file into memory"""

    from fal.upload import BaseMultipartUpload

    # Create a file with multiple chunks
    test_file = Path(temp_dir) / "large.txt"
    test_file.write_bytes(b"x" * (10 * 1024 * 1024 + 100))  # ~10MB

    mock_client = MagicMock()

    initiate_response = MagicMock()
    initiate_response.status_code = 200
    initiate_response.json.return_value = {"upload_id": "test-id"}

    part_response = MagicMock()
    part_response.status_code = 200
    part_response.json.return_value = {"part_number": 1, "etag": "abc"}

    complete_response = MagicMock()
    complete_response.status_code = 200
    complete_response.json.return_value = {"etag": "final"}

    mock_client.request.side_effect = [
        initiate_response,
        part_response,
        part_response,
        complete_response,
    ]

    class TestUpload(BaseMultipartUpload):
        @property
        def initiate_url(self):
            return "/init"

        @property
        def part_url(self):
            return "/part"

        @property
        def complete_url(self):
            return "/complete"

    uploader = TestUpload(
        client=mock_client,
        chunk_size=10 * 1024 * 1024,  # 10MB
        max_concurrency=2,
    )

    # Should not raise MemoryError even with large file
    etag = uploader.upload_file(str(test_file))
    assert etag == "final"


def test_sync_files_with_ignore_patterns(file_sync, temp_dir):
    """Test sync_files properly filters files using ignore patterns"""
    # Create test files
    app_file = Path(temp_dir) / "app.py"
    app_file.write_text("# app file")

    pyc_file = Path(temp_dir) / "test.pyc"
    pyc_file.write_text("bytecode")

    fs = FileSync(str(app_file))

    # Mock server responses to avoid actual network calls
    with patch.object(fs, "check_hashes_on_server", return_value=[]):
        all_files, errors = fs.sync_files(
            paths=[str(temp_dir)], files_ignore=[re.compile(r"\.pyc$")]
        )

        # Should only process app.py, not the ignored .pyc file
        relative_paths = [f.relative_path for f in all_files]

        assert "app.py" in relative_paths
        assert "test.pyc" not in relative_paths
        assert len(errors) == 0


def test_sync_files_basic_workflow(file_sync, temp_dir):
    """Test basic sync_files workflow"""
    test_file = Path(temp_dir) / "test.txt"
    test_file.write_text("content")

    fs = FileSync(str(Path(temp_dir) / "app.py"))

    # Mock server to indicate all files need upload
    with patch.object(fs, "check_hashes_on_server") as mock_check, patch.object(
        fs, "upload_file_multipart", return_value="test_etag"
    ):
        # Make server say all hashes are missing (need upload)
        mock_check.side_effect = lambda hashes: hashes

        all_files, errors = fs.sync_files([str(test_file)])

        assert len(all_files) >= 1
        assert len(errors) == 0


def test_sync_files_deduplication(file_sync, temp_dir):
    """Test that duplicate absolute paths are deduplicated"""
    test_file = Path(temp_dir) / "test.txt"
    test_file.write_text("content")

    fs = FileSync(str(Path(temp_dir) / "app.py"))

    with patch.object(fs, "check_hashes_on_server", return_value=[]):
        all_files, errors = fs.sync_files([str(test_file), str(test_file)])

        # Should only process the file once
        total_files = len(all_files)
        assert total_files == 1
        assert len(errors) == 0


def test_close_client(file_sync):
    """Test HTTP client cleanup"""
    mock_client = MagicMock()

    with patch.object(file_sync, "_client", mock_client):
        file_sync.close()
        mock_client.close.assert_called_once()


def test_multipart_upload_empty_file(temp_dir):
    """Test that multipart upload handles empty files correctly"""
    from fal.upload import AppFileMultipartUpload

    # Create empty file
    test_file = Path(temp_dir) / "__init__.py"
    test_file.write_bytes(b"")

    metadata = file_sync_mod.FileMetadata.from_path(
        test_file, relative="__init__.py", absolute=str(test_file)
    )

    mock_client = MagicMock()

    initiate_response = MagicMock()
    initiate_response.status_code = 200
    initiate_response.json.return_value = {"upload_id": "test-upload-id"}

    part_response = MagicMock()
    part_response.status_code = 200
    part_response.json.return_value = {
        "part_number": 1,
        "etag": "d41d8cd98f00b204e9800998ecf8427e",
    }

    complete_response = MagicMock()
    complete_response.status_code = 200
    complete_response.json.return_value = {"etag": "d41d8cd98f00b204e9800998ecf8427e"}

    mock_client.request.side_effect = [
        initiate_response,
        part_response,
        complete_response,
    ]

    uploader = AppFileMultipartUpload(
        client=mock_client,
        file_hash=metadata.hash,
        metadata=metadata.to_dict(),
    )

    etag = uploader.upload_file(str(test_file))

    assert etag == "d41d8cd98f00b204e9800998ecf8427e"
    assert mock_client.request.call_count == 3  # initiate, part, complete

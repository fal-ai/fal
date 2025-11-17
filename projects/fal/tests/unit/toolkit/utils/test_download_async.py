"""Tests for async download utilities."""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fal.toolkit.utils.download_utils import (
    DownloadError,
    download_file_async,
    download_file_to_dir_async,
    download_model_weights_async,
)


class TestDownloadFileAsync:
    """Tests for download_file_async (lightweight downloader)."""

    @pytest.mark.asyncio
    async def test_download_returns_bytes_when_no_output_path(self):
        """Test that download_file_async returns bytes when output_path is None."""
        mock_response = MagicMock()
        mock_response.content = b"test content"
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )

            result = await download_file_async("http://example.com/file.txt")

            assert isinstance(result, bytes)
            assert result == b"test content"

    @pytest.mark.asyncio
    async def test_download_saves_to_output_path(self):
        """Test that download_file_async saves to file when output_path is provided."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test.txt"

            mock_response = MagicMock()
            mock_response.content = b"test content"
            mock_response.raise_for_status = MagicMock()

            with patch("httpx.AsyncClient") as mock_client:
                mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                    return_value=mock_response
                )

                result = await download_file_async(
                    "http://example.com/file.txt", output_path=output_path
                )

                assert isinstance(result, Path)
                assert result == output_path
                assert output_path.exists()
                assert output_path.read_bytes() == b"test content"

    @pytest.mark.asyncio
    async def test_download_handles_data_uri(self):
        """Test that download_file_async handles data URIs correctly."""
        import base64

        content = b"test content"
        b64_content = base64.b64encode(content).decode()
        data_uri = f"data:text/plain;base64,{b64_content}"

        result = await download_file_async(data_uri)

        assert isinstance(result, bytes)
        assert result == content

    @pytest.mark.asyncio
    async def test_download_retries_on_failure(self):
        """Test that download_file_async retries on HTTP errors."""
        mock_response = MagicMock()
        mock_response.content = b"success"
        mock_response.raise_for_status = MagicMock()

        call_count = 0

        async def mock_get(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                import httpx

                raise httpx.HTTPStatusError(
                    "Server error",
                    request=MagicMock(),
                    response=MagicMock(status_code=500),
                )
            return mock_response

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = mock_get

            result = await download_file_async(
                "http://example.com/file.txt", max_retries=3
            )

            assert result == b"success"
            assert call_count == 2  # Failed once, succeeded on second try

    @pytest.mark.asyncio
    async def test_download_respects_max_size(self):
        """Test that download_file_async respects max_size limit."""
        mock_response = MagicMock()
        mock_response.content = b"x" * 1000  # 1000 bytes
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )

            with pytest.raises(DownloadError, match="exceeds maximum allowed size"):
                await download_file_async(
                    "http://example.com/file.txt", max_size=500  # Limit to 500 bytes
                )


class TestDownloadFileToDirAsync:
    """Tests for download_file_to_dir_async (production downloader)."""

    @pytest.mark.asyncio
    async def test_download_auto_detects_filename(self):
        """Test that download_file_to_dir_async auto-detects filename from URL."""
        import httpx

        with tempfile.TemporaryDirectory() as temp_dir:
            mock_stream_response = MagicMock()
            mock_stream_response.headers = {
                "Content-Length": "12",
                "Content-Type": "text/plain",
            }

            async def mock_aiter_bytes():
                yield b"test content"

            mock_stream_response.aiter_bytes = mock_aiter_bytes
            mock_stream_response.raise_for_status = MagicMock()
            mock_stream_response.__aenter__ = AsyncMock(return_value=mock_stream_response)
            mock_stream_response.__aexit__ = AsyncMock()

            with patch("httpx.AsyncClient") as mock_client:
                mock_async_client = mock_client.return_value.__aenter__.return_value
                # Simulate server not supporting HEAD method
                mock_async_client.head = AsyncMock(
                    side_effect=httpx.HTTPStatusError(
                        "Method not allowed",
                        request=MagicMock(),
                        response=MagicMock(status_code=405),
                    )
                )
                mock_async_client.stream = MagicMock(return_value=mock_stream_response)

                result = await download_file_to_dir_async(
                    "http://example.com/test.txt", temp_dir
                )

                assert isinstance(result, str)
                assert result.endswith("test.txt")
                assert Path(result).exists()

    @pytest.mark.asyncio
    async def test_download_uses_custom_filename(self):
        """Test that download_file_to_dir_async uses custom filename when provided."""
        import httpx

        with tempfile.TemporaryDirectory() as temp_dir:
            mock_stream_response = MagicMock()
            mock_stream_response.headers = {
                "Content-Length": "12",
                "Content-Type": "text/plain",
            }

            async def mock_aiter_bytes():
                yield b"test content"

            mock_stream_response.aiter_bytes = mock_aiter_bytes
            mock_stream_response.raise_for_status = MagicMock()
            mock_stream_response.__aenter__ = AsyncMock(return_value=mock_stream_response)
            mock_stream_response.__aexit__ = AsyncMock()

            with patch("httpx.AsyncClient") as mock_client:
                mock_async_client = mock_client.return_value.__aenter__.return_value
                # Simulate server not supporting HEAD method
                mock_async_client.head = AsyncMock(
                    side_effect=httpx.HTTPStatusError(
                        "Method not allowed",
                        request=MagicMock(),
                        response=MagicMock(status_code=405),
                    )
                )
                mock_async_client.stream = MagicMock(return_value=mock_stream_response)

                result = await download_file_to_dir_async(
                    "http://example.com/original.txt",
                    temp_dir,
                    file_name="custom.txt",
                )

                assert isinstance(result, str)
                assert result.endswith("custom.txt")
                assert Path(result).exists()

    @pytest.mark.asyncio
    async def test_download_skips_existing_file(self):
        """Test that download_file_to_dir_async skips download if file exists with matching size."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create existing file
            existing_file = Path(temp_dir) / "test.txt"
            existing_file.write_bytes(b"test content")

            mock_head_response = MagicMock()
            mock_head_response.headers = {
                "Content-Length": str(len(b"test content")),
                "Content-Type": "text/plain",
            }
            mock_head_response.raise_for_status = MagicMock()

            with patch("httpx.AsyncClient") as mock_client:
                mock_async_client = mock_client.return_value.__aenter__.return_value
                mock_async_client.head = AsyncMock(return_value=mock_head_response)
                # Should not call stream since file exists
                mock_async_client.stream = MagicMock()

                result = await download_file_to_dir_async(
                    "http://example.com/test.txt", temp_dir, force=False
                )

                assert result == str(existing_file)
                # Verify stream was not called (file already existed)
                mock_async_client.stream.assert_not_called()


class TestDownloadModelWeightsAsync:
    """Tests for download_model_weights_async."""

    @pytest.mark.asyncio
    async def test_download_model_weights_uses_hashed_directory(self):
        """Test that download_model_weights_async saves to hashed directory."""
        with patch(
            "fal.toolkit.utils.download_utils.download_file_to_dir_async"
        ) as mock_download:
            mock_download.return_value = "/data/.fal/model_weights/abc123/model.safetensors"

            result = await download_model_weights_async("http://example.com/model.safetensors")

            assert result == "/data/.fal/model_weights/abc123/model.safetensors"
            mock_download.assert_called_once()

            # Verify it uses a hashed directory
            call_args = mock_download.call_args
            target_dir = str(call_args.kwargs["target_dir"])
            assert "/model_weights/" in target_dir

    @pytest.mark.asyncio
    async def test_download_model_weights_lora_max_size(self):
        """Test that download_model_weights_async sets LoRA size limit when lora=True."""
        with patch(
            "fal.toolkit.utils.download_utils.download_file_to_dir_async"
        ) as mock_download:
            mock_download.return_value = "/data/.fal/model_weights/abc123/lora.safetensors"

            await download_model_weights_async(
                "http://example.com/lora.safetensors", lora=True
            )

            call_args = mock_download.call_args
            max_size = call_args.kwargs["max_size"]
            assert max_size == 1750 * 1024 * 1024  # 1.75 GB


@pytest.mark.asyncio
async def test_download_file_async_integration():
    """Integration test - actually download a small file (requires network)."""
    # This test requires network access and might be skipped in CI
    pytest.skip("Skipping network-dependent test")

    # Example of how to test with real download:
    # result = await download_file_async("https://httpbin.org/bytes/1024")
    # assert isinstance(result, bytes)
    # assert len(result) == 1024


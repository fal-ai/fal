from __future__ import annotations

import pytest

from fal.exceptions._cuda import (
    CUDAOutOfMemoryException,
    _is_gpu_error,
    catch_gpu_exceptions,
)


class TestIsGpuError:
    def test_cuda_marker(self):
        exc = RuntimeError("CUDA error: device-side assert triggered")
        assert _is_gpu_error(exc) is True

    def test_cudnn_marker(self):
        exc = RuntimeError("cuDNN error: CUDNN_STATUS_INTERNAL_ERROR")
        assert _is_gpu_error(exc) is True

    def test_nvml_marker(self):
        exc = RuntimeError("NVML_SUCCESS == r INTERNAL ASSERT FAILED")
        assert _is_gpu_error(exc) is True

    def test_case_insensitive(self):
        exc = RuntimeError("CuDa something went wrong")
        assert _is_gpu_error(exc) is True

    def test_non_gpu_error(self):
        exc = RuntimeError("file not found")
        assert _is_gpu_error(exc) is False

    def test_empty_message(self):
        exc = RuntimeError("")
        assert _is_gpu_error(exc) is False


class TestCatchGpuExceptionsContextManager:
    def test_no_exception(self):
        with catch_gpu_exceptions():
            pass

    def test_cuda_oom_pattern(self):
        with pytest.raises(CUDAOutOfMemoryException):
            with catch_gpu_exceptions():
                raise RuntimeError("CUDA error: out of memory")

    def test_cudnn_error(self):
        with pytest.raises(CUDAOutOfMemoryException):
            with catch_gpu_exceptions():
                raise RuntimeError("cuDNN error: CUDNN_STATUS_NOT_SUPPORTED.")

    def test_broad_gpu_marker(self):
        with pytest.raises(CUDAOutOfMemoryException) as exc_info:
            with catch_gpu_exceptions():
                raise RuntimeError("CUDA error: device-side assert triggered")
        assert exc_info.value.message == "GPU error"

    def test_non_gpu_passthrough(self):
        with pytest.raises(ValueError, match="not a gpu error"):
            with catch_gpu_exceptions():
                raise ValueError("not a gpu error")

    def test_cuda_oom_exception_passthrough(self):
        """CUDAOutOfMemoryException should pass through without double-wrapping."""
        original = CUDAOutOfMemoryException()
        with pytest.raises(CUDAOutOfMemoryException) as exc_info:
            with catch_gpu_exceptions():
                raise original
        assert exc_info.value is original

    def test_exception_chain_preserved(self):
        with pytest.raises(CUDAOutOfMemoryException) as exc_info:
            with catch_gpu_exceptions():
                raise RuntimeError("CUDA error: out of memory")
        assert isinstance(exc_info.value.__cause__, RuntimeError)

    def test_nvml_broad_marker(self):
        with pytest.raises(CUDAOutOfMemoryException) as exc_info:
            with catch_gpu_exceptions():
                raise OSError("NVML driver failure")
        assert exc_info.value.message == "GPU error"


class TestCatchGpuExceptionsDecorator:
    def test_clean_function(self):
        @catch_gpu_exceptions()
        def clean():
            return 42

        assert clean() == 42

    def test_gpu_error_catch(self):
        @catch_gpu_exceptions()
        def bad_gpu():
            raise RuntimeError("CUDA error: out of memory")

        with pytest.raises(CUDAOutOfMemoryException):
            bad_gpu()

    def test_non_gpu_passthrough(self):
        @catch_gpu_exceptions()
        def bad_logic():
            raise TypeError("wrong type")

        with pytest.raises(TypeError, match="wrong type"):
            bad_logic()

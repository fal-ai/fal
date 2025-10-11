import pytest
from fal.distributed.utils import (
    format_for_serialization,
    distributed_serialize,
    distributed_deserialize,
    encode_text_event,
    KeepAliveTimer,
    has_type_name,
    is_torch_tensor,
    is_numpy_array,
    is_pil_image,
)
import numpy as np
from PIL import Image
import threading

def test_format_for_serialization_numpy():
    array = np.array([1, 2, 3])
    serialized = format_for_serialization(array)
    assert serialized["content_type"] == "application/ndarray"
    assert serialized["shape"] == array.shape

def test_format_for_serialization_pil():
    image = Image.new("RGB", (10, 10))
    serialized = format_for_serialization(image)
    assert serialized["content_type"].startswith("image/")

def test_distributed_serialize_deserialize():
    data = {"key": "value"}
    serialized = distributed_serialize(data)
    deserialized = distributed_deserialize(serialized)
    assert deserialized == data

def test_encode_text_event():
    data = {"key": "value"}
    encoded = encode_text_event(data)
    assert encoded.startswith(b"data: ")

def test_keepalive_timer():
    calls = []
    def callback():
        calls.append(1)
    timer = KeepAliveTimer(callback, 0.1, start=True)
    threading.Event().wait(0.2)
    timer.cancel()
    assert len(calls) == 1

def test_has_type_name():
    class Dummy:
        pass
    assert has_type_name(Dummy(), "Dummy")

def test_is_torch_tensor():
    # Skip if torch is not installed
    try:
        import torch
        tensor = torch.tensor([1, 2, 3])
        assert is_torch_tensor(tensor)
    except ImportError:
        pytest.skip("torch not installed")

def test_is_numpy_array():
    assert is_numpy_array(np.array([1, 2, 3]))

def test_is_pil_image():
    image = Image.new("RGB", (10, 10))
    assert is_pil_image(image)


def test_serialization_roundtrip_complex():
    """Test nested dictionaries and lists serialize correctly."""
    data = {
        "nested": {"list": [1, 2, 3], "dict": {"key": "value"}},
        "key": "value",
        "array": [{"a": 1}, {"b": 2}]
    }
    serialized = distributed_serialize(data)
    deserialized = distributed_deserialize(serialized)
    assert deserialized == data


def test_keepalive_timer_cancel_before_trigger():
    """Test timer can be cancelled before it fires."""
    calls = []
    timer = KeepAliveTimer(lambda: calls.append(1), 10, start=True)
    timer.cancel()
    import time
    time.sleep(0.1)
    assert len(calls) == 0


def test_format_for_serialization_none():
    """Test None values pass through correctly."""
    assert format_for_serialization(None) is None


def test_serialization_with_mixed_types():
    """Test serialization handles mixed types (PIL images, numpy, dicts)."""
    data = {
        "image": Image.new("RGB", (5, 5)),
        "array": np.array([1, 2, 3]),
        "text": "hello",
        "number": 42
    }
    serialized = format_for_serialization(data)
    
    assert serialized["image"]["content_type"].startswith("image/")
    assert serialized["array"]["content_type"] == "application/ndarray"
    assert serialized["text"] == "hello"
    assert serialized["number"] == 42


def test_encode_text_event_format():
    """Test that text events are properly formatted as SSE."""
    data = {"status": "processing"}
    encoded = encode_text_event(data)
    
    assert encoded.startswith(b"data: ")
    assert encoded.endswith(b"\n\n")
    
    # Verify it's valid JSON
    import json
    json_part = encoded[len(b"data: "):-2]
    parsed = json.loads(json_part)
    assert parsed["status"] == "processing"


def test_keepalive_timer_reset():
    """Test that resetting the timer works correctly."""
    calls = []
    timer = KeepAliveTimer(lambda: calls.append(1), 0.05, start=True)
    import time
    time.sleep(0.03)  # Wait but don't let it fire
    timer.reset()  # Reset should cancel and restart
    time.sleep(0.03)  # Still shouldn't fire (reset extended time)
    assert len(calls) == 0
    timer.cancel()
    time.sleep(0.1)  # Make sure it doesn't fire after cancel
    assert len(calls) == 0

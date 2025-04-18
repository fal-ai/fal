import pytest
from datetime import datetime, timezone
from typing import Any, Optional
from fal_client.client import LogMessage


def test_pydantic_v1_compatibility():
    """Test compatibility with pydantic v1."""
    try:
        from pydantic import BaseModel
    except ImportError:
        pytest.skip("pydantic v1 not installed")

    class PydanticLogMessage(BaseModel):
        timestamp: datetime
        level: str
        message: str
        source: str
        metadata: Optional[dict[str, Any]] = None

    # Test creating from LogMessage
    log_msg = LogMessage(
        timestamp=datetime.now(timezone.utc),
        level="INFO",
        message="test",
        source="test_source",
    )
    pydantic_msg = PydanticLogMessage(
        timestamp=log_msg.timestamp,
        level=log_msg.level,
        message=log_msg.message,
        source=log_msg.source,
    )
    assert pydantic_msg.level == "INFO"
    assert pydantic_msg.message == "test"
    assert pydantic_msg.source == "test_source"
    assert isinstance(pydantic_msg.timestamp, datetime)

    # Test creating from dict
    data = {
        "timestamp": "2024-01-01T00:00:00+00:00",
        "level": "WARNING",
        "message": "test warning",
        "source": "test_source",
        "metadata": {"key": "value"},
    }
    log_msg = LogMessage.from_dict(data)
    pydantic_msg = PydanticLogMessage(**data)
    assert pydantic_msg.level == log_msg.level
    assert pydantic_msg.message == log_msg.message
    assert pydantic_msg.source == log_msg.source
    assert pydantic_msg.timestamp == log_msg.timestamp
    assert pydantic_msg.metadata == log_msg.metadata


def test_pydantic_v2_compatibility():
    """Test compatibility with pydantic v2."""
    try:
        from pydantic import BaseModel, field_validator
    except ImportError:
        pytest.skip("pydantic v2 not installed")

    class PydanticV2LogMessage(BaseModel):
        timestamp: datetime
        level: str
        message: str
        source: str
        metadata: Optional[dict[str, Any]] = None

        @field_validator("level")
        def validate_level(cls, v):
            v = v.upper()
            valid_levels = ["INFO", "WARNING", "ERROR", "DEBUG"]
            if v not in valid_levels:
                return "INFO"
            return v

    # Test creating from LogMessage
    log_msg = LogMessage(
        timestamp=datetime.now(timezone.utc),
        level="INFO",
        message="test",
        source="test_source",
    )
    pydantic_msg = PydanticV2LogMessage(
        timestamp=log_msg.timestamp,
        level=log_msg.level,
        message=log_msg.message,
        source=log_msg.source,
    )
    assert pydantic_msg.level == "INFO"
    assert pydantic_msg.message == "test"
    assert pydantic_msg.source == "test_source"
    assert isinstance(pydantic_msg.timestamp, datetime)

    # Test creating from dict
    data = {
        "timestamp": "2024-01-01T00:00:00+00:00",
        "level": "WARNING",
        "message": "test warning",
        "source": "test_source",
        "metadata": {"key": "value"},
    }
    log_msg = LogMessage.from_dict(data)
    pydantic_msg = PydanticV2LogMessage(**data)
    assert pydantic_msg.level == log_msg.level
    assert pydantic_msg.message == log_msg.message
    assert pydantic_msg.source == log_msg.source
    assert pydantic_msg.timestamp == log_msg.timestamp
    assert pydantic_msg.metadata == log_msg.metadata

    # Test invalid level handling
    pydantic_msg = PydanticV2LogMessage(
        timestamp=datetime.now(timezone.utc),
        level="invalid",
        message="test",
        source="test_source",
    )
    assert pydantic_msg.level == "INFO"  # Should default to INFO


def test_pydantic_roundtrip():
    """Test roundtrip conversion between LogMessage and pydantic models."""
    try:
        from pydantic import BaseModel
    except ImportError:
        pytest.skip("pydantic not installed")

    class PydanticLogMessage(BaseModel):
        timestamp: datetime
        level: str
        message: str
        source: str
        metadata: Optional[dict[str, Any]] = None

    # Create LogMessage
    log_msg = LogMessage(
        timestamp=datetime.now(timezone.utc),
        level="INFO",
        message="test",
        source="test_source",
    )

    # Convert to pydantic
    pydantic_msg = PydanticLogMessage(
        timestamp=log_msg.timestamp,
        level=log_msg.level,
        message=log_msg.message,
        source=log_msg.source,
    )

    # Convert back to dict (using model_dump for pydantic v2)
    try:
        data = pydantic_msg.model_dump()
    except AttributeError:
        data = pydantic_msg.dict()

    # Create new LogMessage from dict
    new_log_msg = LogMessage.from_dict(data)

    # Compare
    assert new_log_msg.level == log_msg.level
    assert new_log_msg.message == log_msg.message
    assert new_log_msg.source == log_msg.source
    # Note: timestamp comparison might be tricky due to timezone handling
    assert abs((new_log_msg.timestamp - log_msg.timestamp).total_seconds()) < 1

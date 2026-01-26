"""Integration tests for environment management using the API layer."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest

from fal.api import FalServerlessError, SyncServerlessClient


@pytest.fixture(scope="function")
def test_env_name():
    """Generate a unique environment name for testing."""
    return f"test-env-{uuid.uuid4().hex[:8]}"


@pytest.fixture(scope="function")
def client():
    """Create a SyncServerlessClient for API layer testing."""
    import os

    host = os.getenv("FAL_HOST")
    api_key = os.getenv("FAL_KEY")
    return SyncServerlessClient(host=host, api_key=api_key)


@pytest.mark.flaky(max_runs=3)
def test_environment_lifecycle(client: SyncServerlessClient, test_env_name: str):
    """Test creating, listing, and deleting environments."""
    # Create environment
    env = client.environments.create(
        test_env_name, description="Integration test environment"
    )
    assert env.name == test_env_name
    assert env.description == "Integration test environment"
    assert env.created_at is not None

    try:
        # List environments - verify our env exists
        environments = client.environments.list()
        env_names = [e.name for e in environments]
        assert test_env_name in env_names

        # Find our environment in the list
        our_env = next(e for e in environments if e.name == test_env_name)
        assert our_env.description == "Integration test environment"

    finally:
        # Cleanup - delete environment
        client.environments.delete(test_env_name)

        # Verify deletion
        environments_after = client.environments.list()
        env_names_after = [e.name for e in environments_after]
        assert test_env_name not in env_names_after


@pytest.mark.flaky(max_runs=3)
def test_environment_with_secrets(client: SyncServerlessClient, test_env_name: str):
    """Test that secrets are scoped to environments."""
    secret_name = f"TEST_SECRET_{uuid.uuid4().hex[:8]}"

    # Create test environment
    client.environments.create(test_env_name, description="Test env for secrets")

    try:
        # Set a secret in the test environment
        client.secrets.set(secret_name, "test_value", environment_name=test_env_name)

        # List secrets in test environment
        secrets_in_env = client.secrets.list(environment_name=test_env_name)
        secret_names = [s.name for s in secrets_in_env]
        assert secret_name in secret_names

        # Verify secret has correct environment
        our_secret = next(s for s in secrets_in_env if s.name == secret_name)
        assert our_secret.environment_name == test_env_name

        # Delete the secret before deleting the environment
        client.secrets.unset(secret_name, environment_name=test_env_name)

        # Delete the environment
        client.environments.delete(test_env_name)

        # Verify the environment is gone
        environments = client.environments.list()
        assert test_env_name not in [e.name for e in environments]

    except Exception:
        # Cleanup on failure
        try:
            client.secrets.unset(secret_name, environment_name=test_env_name)
        except Exception:
            pass
        try:
            client.environments.delete(test_env_name)
        except Exception:
            pass
        raise


def test_delete_nonexistent_environment(client: SyncServerlessClient):
    """Test deleting an environment that doesn't exist raises FalServerlessError."""
    with pytest.raises(FalServerlessError):
        client.environments.delete(f"nonexistent-env-{uuid.uuid4().hex[:8]}")


def test_create_duplicate_environment(client: SyncServerlessClient, test_env_name: str):
    """Test creating an environment with a duplicate name raises FalServerlessError."""
    # Create first environment
    client.environments.create(test_env_name)

    try:
        # Attempt to create duplicate - should raise FalServerlessError
        with pytest.raises(FalServerlessError):
            client.environments.create(test_env_name)
    finally:
        # Cleanup
        client.environments.delete(test_env_name)


@pytest.mark.flaky(max_runs=3)
def test_list_environments_returns_list(client: SyncServerlessClient):
    """Test that list_environments returns a valid list."""
    environments = client.environments.list()
    assert isinstance(environments, list)


@pytest.mark.flaky(max_runs=3)
def test_environment_created_at_timestamp(
    client: SyncServerlessClient, test_env_name: str
):
    """Test that created_at timestamp is properly set."""
    # Create environment
    env = client.environments.create(test_env_name)

    try:
        assert env.created_at is not None
        # Check that it's a datetime object
        assert isinstance(env.created_at, datetime)
        # Check that it's recent (within last minute)
        now = datetime.now(timezone.utc)
        time_diff = (now - env.created_at).total_seconds()
        assert -60 <= time_diff < 60, f"Created time seems wrong: {time_diff}s ago"

    finally:
        client.environments.delete(test_env_name)

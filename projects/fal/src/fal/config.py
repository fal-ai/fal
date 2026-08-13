from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

SETTINGS_SECTION = "__internal__"

NO_PROFILE_ERROR = ValueError(
    "No profile set. Use command [bold]fal profile set[/] to set a profile."
)


class Config:
    _config: dict[str, Any]
    _profile: str | None
    _editing: bool = False

    DEFAULT_CONFIG_PATH = "~/.fal/config.toml"

    def __init__(self, *, validate_profile: bool = False, profile: str | None = None):
        import tomli  # noqa: PLC0415

        self.config_path = os.path.expanduser(
            os.getenv("FAL_CONFIG_PATH", self.DEFAULT_CONFIG_PATH)
        )

        try:
            with open(self.config_path, "rb") as file:
                self._config = tomli.load(file)
        except FileNotFoundError:
            self._config = {}

        selected_profile = (
            profile
            if profile is not None
            else os.getenv("FAL_PROFILE") or self.get_internal("profile")
        )

        # Try to set the profile, but don't fail if it doesn't exist
        try:
            self.profile = selected_profile
        except ValueError:
            # Profile doesn't exist, set to None
            self._profile = None

        if validate_profile and not self.profile:
            raise NO_PROFILE_ERROR

    @property
    def profile(self) -> str | None:
        return self._profile

    @profile.setter
    def profile(self, value: str | None) -> None:
        if value and value not in self.profiles():
            # Don't automatically create profiles - they should be created explicitly
            raise ValueError(
                f"Profile '{value}' does not exist. Create it first or use the profile set command."  # noqa: E501
            )
        elif not value:
            self.unset_internal("profile")

        self._profile = value

    def profiles(self) -> list[str]:
        keys: list[str] = []
        for key, value in self._config.items():
            if key != SETTINGS_SECTION and isinstance(value, dict):
                keys.append(key)

        return keys

    def save(self) -> None:
        import tomli_w  # noqa: PLC0415

        with open(self.config_path, "wb") as file:
            tomli_w.dump(self._config, file)

    def get(self, key: str) -> str | None:
        if not self.profile:
            return None

        return self._config.get(self.profile, {}).get(key)

    def set(self, key: str, value: str) -> None:
        if not self.profile:
            raise NO_PROFILE_ERROR

        self._config[self.profile][key] = value

    def unset(self, key: str) -> None:
        if not self.profile:
            raise NO_PROFILE_ERROR

        self._config.get(self.profile, {}).pop(key, None)

    def get_internal(self, key: str) -> str | None:
        if SETTINGS_SECTION not in self._config:
            self._config[SETTINGS_SECTION] = {}

        return self._config[SETTINGS_SECTION].get(key)

    def get_global(self, key: str) -> Any | None:
        return self._config.get(key)

    def set_internal(self, key: str, value: str | None) -> None:
        if SETTINGS_SECTION not in self._config:
            self._config[SETTINGS_SECTION] = {}

        if value is None:
            del self._config[SETTINGS_SECTION][key]
        else:
            self._config[SETTINGS_SECTION][key] = value

    def unset_internal(self, key: str) -> None:
        self._config.get(SETTINGS_SECTION, {}).pop(key, None)

    def delete_profile(self, profile: str) -> None:
        if profile not in self.profiles():
            raise ValueError(f"Profile '{profile}' does not exist.")

        del self._config[profile]

    @contextmanager
    def edit(self) -> Iterator[Config]:
        if self._editing:
            # no-op
            yield self
        else:
            self._editing = True
            yield self
            self.save()
            self._editing = False

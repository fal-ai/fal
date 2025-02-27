import os
from typing import Optional

import tomli
import tomli_w

SETTINGS_SECTION = "__internal__"


class Config:
    _config: dict[str, dict[str, str]]
    _profile: str

    DEFAULT_CONFIG_PATH = "~/.fal/config.toml"
    DEFAULT_PROFILE = "default"

    def __init__(self):
        self.config_path = os.path.expanduser(
            os.getenv("FAL_CONFIG_PATH", self.DEFAULT_CONFIG_PATH)
        )

        try:
            with open(self.config_path, "rb") as file:
                self._config = tomli.load(file)
        except FileNotFoundError:
            self._config = {}

        profile = os.getenv("FAL_PROFILE")
        if not profile:
            profile = self.get_internal("profile")
        if not profile:
            profile = self.DEFAULT_PROFILE

        self._profile = profile

    @property
    def profile(self):
        return self._profile

    def profiles(self):
        keys = []
        for key in self._config:
            if key != SETTINGS_SECTION:
                keys.append(key)

        return keys

    def save(self):
        with open(self.config_path, "wb") as file:
            tomli_w.dump(self._config, file)

    def get(self, key: str) -> Optional[str]:
        return self._config.get(self._profile, {}).get(key)

    def set(self, key: str, value: str):
        if self._profile not in self._config:
            self._config[self._profile] = {}

        self._config[self._profile][key] = value

    def get_internal(self, key):
        if SETTINGS_SECTION not in self._config:
            self._config[SETTINGS_SECTION] = {}

        return self._config[SETTINGS_SECTION].get(key)

    def set_internal(self, key, value):
        if SETTINGS_SECTION not in self._config:
            self._config[SETTINGS_SECTION] = {}

        self._config[SETTINGS_SECTION][key] = value

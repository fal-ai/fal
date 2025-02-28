import os
from typing import Dict, List, Optional

SETTINGS_SECTION = "__internal__"


class Config:
    _config: Dict[str, Dict[str, str]]
    _profile: Optional[str]

    DEFAULT_CONFIG_PATH = "~/.fal/config.toml"

    def __init__(self):
        import tomli

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

        self.profile = profile

    @property
    def profile(self) -> Optional[str]:
        return self._profile

    @profile.setter
    def profile(self, value: Optional[str]) -> None:
        if value and value not in self._config:
            self._config[value] = {}

        self._profile = value

    def profiles(self) -> List[str]:
        keys = []
        for key in self._config:
            if key != SETTINGS_SECTION:
                keys.append(key)

        return keys

    def save(self) -> None:
        import tomli_w

        with open(self.config_path, "wb") as file:
            tomli_w.dump(self._config, file)

    def get(self, key: str) -> Optional[str]:
        if not self.profile:
            return None

        return self._config.get(self.profile, {}).get(key)

    def set(self, key: str, value: str) -> None:
        if not self.profile:
            raise ValueError("No profile set.")

        self._config[self.profile][key] = value

    def get_internal(self, key: str) -> Optional[str]:
        if SETTINGS_SECTION not in self._config:
            self._config[SETTINGS_SECTION] = {}

        return self._config[SETTINGS_SECTION].get(key)

    def set_internal(self, key: str, value: Optional[str]) -> None:
        if SETTINGS_SECTION not in self._config:
            self._config[SETTINGS_SECTION] = {}

        if value is None:
            del self._config[SETTINGS_SECTION][key]
        else:
            self._config[SETTINGS_SECTION][key] = value

    def delete(self, profile: str) -> None:
        del self._config[profile]

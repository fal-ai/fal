import os

import tomli


class Config:
    DEFAULT_CONFIG_PATH = "~/.fal/config.toml"
    DEFAULT_PROFILE = "default"

    def __init__(self):
        self.config_path = os.path.expanduser(
            os.getenv("FAL_CONFIG_PATH", self.DEFAULT_CONFIG_PATH)
        )
        self.profile = os.getenv("FAL_PROFILE", self.DEFAULT_PROFILE)

        try:
            with open(self.config_path, "rb") as file:
                self.config = tomli.load(file)
        except FileNotFoundError:
            self.config = {}

    def get(self, key):
        return self.config.get(self.profile, {}).get(key)

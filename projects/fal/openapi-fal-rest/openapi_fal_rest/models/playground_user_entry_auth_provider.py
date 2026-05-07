from enum import Enum


class PlaygroundUserEntryAuthProvider(str, Enum):
    AUTH0 = "auth0"
    WORKOS = "workos"

    def __str__(self) -> str:
        return str(self.value)

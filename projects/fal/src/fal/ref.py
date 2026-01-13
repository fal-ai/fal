from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from fal.app import App

# Global reference to the currently running app
# This is useful for accessing the app in runtime
current_app: Optional["App"] = None  # type: ignore[assignment]


def set_current_app(app: "App"):
    global current_app  # noqa: PLW0603
    assert current_app is None, "current_app is already set"
    current_app = app


def get_current_app() -> Optional["App"]:
    return current_app

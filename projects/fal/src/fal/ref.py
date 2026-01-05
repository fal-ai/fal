from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fal.app import App

# Global reference to the currently running app
# This is useful for accessing the app in runtime
current_app: "App" | None = None  # type: ignore[assignment]


def set_current_app(app: "App"):
    global current_app  # noqa: PLW0603
    assert current_app is None, "current_app is already set"
    current_app = app


def get_current_app() -> "App" | None:
    return current_app

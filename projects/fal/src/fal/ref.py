from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fal.app import App

# Global reference to the currently running app
# This is useful for accessing the app in runtime
current_app: "App" = None  # type: ignore[assignment]


def set_current_app(app: "App"):
    global current_app  # noqa: PLW0603
    current_app = app


def get_current_app() -> "App":
    return current_app

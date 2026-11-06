"""The error type `fal dev` raises for failures it can explain to the user."""


class DevError(Exception):
    """A `fal dev` failure; the message is shown to the user as is."""

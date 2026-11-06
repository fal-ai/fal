"""The account a `fal dev` box runs as."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Account:
    # The account's name. `fal_team` is what fal is given: None under an API key.
    team: str | None
    personal: bool
    via_key: bool = False

    @property
    def fal_team(self) -> str | None:
        """The team to give fal: none under an API key, which names its account."""
        return None if self.via_key else self.team

    @property
    def runs_as(self) -> str:
        if self.personal:
            return "your personal account"
        if self.via_key:
            return f"the API key's account ({self.team or 'unnamed'})"
        return f"team {self.team}"

    def __str__(self) -> str:
        if self.personal:
            return f"personal ({self.team})"
        if self.via_key:
            return f"{self.team or 'unnamed account'} (API key)"
        return str(self.team)

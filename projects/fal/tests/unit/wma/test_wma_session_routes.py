"""App-authored contracts stay associated with their session handlers."""

import pydantic
import pytest

if not hasattr(pydantic, "TypeAdapter"):
    pytest.skip("realtime contracts require Pydantic 2", allow_module_level=True)

from fal import endpoint
from fal.wma import App, MediaContract, RealtimeContract, Track


class TwoWorlds(App):
    realtime_contract = RealtimeContract(
        media=MediaContract(send=(Track(kind="video", source="camera"),))
    )
    realtime_contracts = {
        "/game/start-session": RealtimeContract(
            media=MediaContract(receive=(Track(kind="video"),))
        )
    }

    @endpoint("/game/start-session")
    async def game(self) -> dict:
        return {}


def test_metadata_binds_each_route_to_its_own_standalone_contract():
    metadata = TwoWorlds.build_metadata()
    spec = metadata["openapi"]
    primary = metadata["asyncapi"]
    game = primary["x-fal-contracts"]["/game/start-session"]
    assert (
        primary["x-fal-openapi"]["operationId"]
        == spec["paths"]["/start-session"]["post"]["operationId"]
    )
    assert (
        game["x-fal-openapi"]["operationId"]
        == spec["paths"]["/game/start-session"]["post"]["operationId"]
    )
    assert (
        spec["paths"]["/game/start-session"]["x-fal-realtime"]["asyncapi"]["url"]
        == "./asyncapi.json#/x-fal-contracts/~1game~1start-session"
    )
    assert primary["x-fal-media"]["send"]
    assert not game["x-fal-media"]["send"]
    assert game["x-fal-media"]["receive"]
    assert "x-fal-contracts" not in game


def test_single_custom_route_does_not_advertise_default_handler():
    class GameOnly(TwoWorlds):
        realtime_contract = None

    metadata = GameOnly.build_metadata()
    assert "x-fal-realtime" not in metadata["openapi"]["paths"]["/start-session"]
    assert "x-fal-contracts" not in metadata["asyncapi"]
    assert (
        metadata["asyncapi"]["x-fal-openapi"]["operationId"]
        == metadata["openapi"]["paths"]["/game/start-session"]["post"]["operationId"]
    )


def test_missing_handler_fails_during_metadata_generation():
    class Missing(App):
        realtime_contracts = {"/missing": RealtimeContract()}

    with pytest.raises(ValueError, match="POST /missing"):
        Missing.build_metadata()


@pytest.mark.parametrize(
    "path", ["game", "/../game", "/game?x=1", "/game/", "/a%2fb", "/a//b"]
)
def test_paths_that_bridge_cannot_route_are_rejected(path):
    class Invalid(App):
        realtime_contracts = {path: RealtimeContract()}

    with pytest.raises(ValueError, match="session path"):
        Invalid.build_metadata()


def test_duplicate_default_declaration_is_rejected():
    class Ambiguous(TwoWorlds):
        realtime_contracts = {"/start-session": RealtimeContract()}

    with pytest.raises(ValueError, match="both"):
        Ambiguous.build_metadata()


def test_contract_mapping_does_not_mutate_parent():
    before = dict(TwoWorlds.realtime_contracts)
    TwoWorlds.build_metadata()
    TwoWorlds.build_metadata()
    assert TwoWorlds.realtime_contracts == before


def test_duplicate_operation_ids_are_rejected():
    spec = TwoWorlds(_allow_init=True).openapi()
    spec["paths"]["/game/start-session"]["post"]["operationId"] = spec["paths"][
        "/start-session"
    ]["post"]["operationId"]
    with pytest.raises(ValueError, match="distinct operationIds"):
        TwoWorlds._render_asyncapi(spec)


def test_root_session_handler_can_publish_a_contract():
    class Root(App):
        realtime_contracts = {"/": RealtimeContract()}

        @endpoint("/")
        async def root(self) -> dict:
            return {}

    metadata = Root.build_metadata()
    assert (
        metadata["openapi"]["paths"]["/"]["x-fal-realtime"]["asyncapi"]["url"]
        == "./asyncapi.json"
    )

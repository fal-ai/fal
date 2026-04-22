from __future__ import annotations

import isolate_proto


def test_proto():
    test_mod = isolate_proto.MachineRequirements(machine_type="XL")
    assert test_mod.machine_type == "XL"


def test_callable_entrypoint_fields():
    hosted_map = isolate_proto.HostedMap(
        entrypoint="pkg.mod:func",
        run_on_main_thread=True,
    )
    assert hosted_map.WhichOneof("callable") == "entrypoint"
    assert hosted_map.entrypoint == "pkg.mod:func"
    assert hosted_map.run_on_main_thread is True

    hosted_run = isolate_proto.HostedRun(
        entrypoint="pkg.mod:func",
        run_on_main_thread=True,
    )
    assert hosted_run.WhichOneof("callable") == "entrypoint"
    assert hosted_run.entrypoint == "pkg.mod:func"
    assert hosted_run.run_on_main_thread is True

    register_request = isolate_proto.RegisterApplicationRequest(
        entrypoint="pkg.mod:App.run",
        run_on_main_thread=True,
    )
    assert register_request.WhichOneof("callable") == "entrypoint"
    assert register_request.entrypoint == "pkg.mod:App.run"
    assert register_request.run_on_main_thread is True

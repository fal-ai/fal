from __future__ import annotations

import isolate_proto


def test_proto():
    test_mod = isolate_proto.MachineRequirements("XL")
    assert test_mod.machine_type == "XL"

from __future__ import annotations

from fal import cached, function


class DepCass:
    def dep1(self):
        return "1"


def dep2():
    return "2"


@cached
def dep3():
    return "3"


@function("virtualenv", keep_alive=0, machine_type="XS")
def mainified():
    return DepCass().dep1() + dep2() + dep3()

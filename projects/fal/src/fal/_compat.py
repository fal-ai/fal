import sys


def removeprefix(s: str, prefix: str) -> str:
    if sys.version_info > (3, 8):
        return s.removeprefix(prefix)

    if s.startswith(prefix):
        return s[len(prefix) :]
    return s

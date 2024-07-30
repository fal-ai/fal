from pathlib import Path

CURR_DIR = Path(__file__).resolve().parent


def get_requirements():
    with open(CURR_DIR / "requirements.txt") as fp:
        requirements = fp.read().split()
    return requirements

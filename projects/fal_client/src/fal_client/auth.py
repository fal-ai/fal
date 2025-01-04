import os


class MissingCredentialsError(Exception):
    pass


FAL_RUN_HOST = os.environ.get("FAL_RUN_HOST", "fal.run")


def fetch_credentials() -> str:
    if key := os.getenv("FAL_KEY"):
        return key
    elif (key_id := os.getenv("FAL_KEY_ID")) and (
        fal_key_secret := os.getenv("FAL_KEY_SECRET")
    ):
        return f"{key_id}:{fal_key_secret}"
    else:
        raise MissingCredentialsError(
            "Please set the FAL_KEY environment variable to your API key, or set the FAL_KEY_ID and FAL_KEY_SECRET environment variables."
        )

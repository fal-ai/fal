import json
from typing import Any, Dict, Optional
from urllib.error import HTTPError
from urllib.request import Request

from fal.toolkit.exceptions import KVStoreException
from fal.toolkit.file.providers.fal import _maybe_retry_request, fal_v3_token_manager

FAL_KV_HOST = "https://kv.fal.media"


class KVStore:
    """A key-value store client for interacting with the FAL KV service.

    Args:
        db_name: The name of the database/namespace to use for this KV store.
    """

    def __init__(self, db_name: str):
        self.db_name = db_name

    @property
    def auth_headers(self) -> Dict[str, str]:
        token = fal_v3_token_manager.get_token()
        return {
            "Authorization": f"{token.token_type} {token.token}",
            "User-Agent": "fal/0.1.0",
        }

    def get(self, key: str) -> Optional[str]:
        """Retrieve a value from the key-value store.

        Args:
            key: The key to retrieve the value for.

        Returns:
            The value associated with the key, or None if the key doesn't exist.
        """
        response = self._send_request(
            method="GET",
            path=f"/get/{self.db_name}/{key}",
        )

        if response is None:
            return None

        return response["value"]

    def set(self, key: str, value: str) -> None:
        """Store a value in the key-value store.

        Args:
            key: The key to store the value under.
            value: The value to store.
        """
        self._send_request(
            method="PUT",
            path=f"/set/{self.db_name}/{key}",
            data=value.encode(),
        )

    def _send_request(
        self,
        method: str,
        path: str,
        data: Optional[bytes] = None,
    ) -> Optional[Dict[str, Any]]:
        headers = {
            **self.auth_headers,
            "Accept": "application/json",
        }

        url = FAL_KV_HOST + path
        request = Request(url, headers=headers, method=method, data=data)
        try:
            with _maybe_retry_request(request) as response:
                result = json.load(response)
        except HTTPError as e:
            if e.status == 404:
                return None
            raise KVStoreException(
                f"Error sending request. Status {e.status}: {e.reason}"
            )

        return result

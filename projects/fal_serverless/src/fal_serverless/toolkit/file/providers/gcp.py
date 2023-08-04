from __future__ import annotations

import os
from dataclasses import dataclass, field

from fal_serverless.toolkit import mainify
from fal_serverless.toolkit.file.types import FileData, FileRepository

DEFAULT_URL_TIMEOUT = 60 * 15  # 15 minutes


@mainify
@dataclass
class GoogleStorageRepository(FileRepository):

    url_expiration: int | None = field(default_factory=lambda: DEFAULT_URL_TIMEOUT)
    bucket_name: str | None = field(default="fal_file_storage")
    gcp_account_json: str | None = None

    _storage_client = None

    def __post_init__(self):
        gcp_account_json = self.gcp_account_json
        if gcp_account_json is None:
            gcp_account_json = os.environ.get("GCLOUD_SA_JSON")
            if gcp_account_json is None:
                raise Exception("GCLOUD_SA_JSON environment secret is not set")

        import json

        gcp_account_info = json.loads(gcp_account_json)

        from google.cloud.storage import Client

        self._storage_client = Client.from_service_account_info(gcp_account_info)

    def save(self, data: FileData) -> str:
        import datetime
        from typing import cast

        from google.cloud.storage import Client

        if not isinstance(self._storage_client, Client):
            raise Exception("Google Storage client is not initialized")

        client = cast(Client, self._storage_client)
        bucket = client.get_bucket(self.bucket_name)
        gcp_blob = bucket.blob(data.file_name)
        gcp_blob.upload_from_string(data.data, content_type=data.content_type)

        if self.url_expiration is None:
            return gcp_blob.public_url

        return gcp_blob.generate_signed_url(
            version="v4",
            expiration=datetime.timedelta(minutes=self.url_expiration),
            method="GET",
        )

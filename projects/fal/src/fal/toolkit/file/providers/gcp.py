from __future__ import annotations

import datetime
import json
import os
import posixpath
import uuid
from dataclasses import dataclass

from fal.toolkit.file.types import FileData, FileRepository
from fal.toolkit.utils.retry import retry

DEFAULT_URL_TIMEOUT = 60 * 15  # 15 minutes


@dataclass
class GoogleStorageRepository(FileRepository):
    bucket_name: str = "fal_file_storage"
    url_expiration: int | None = DEFAULT_URL_TIMEOUT
    gcp_account_json: str | None = None
    folder: str = ""

    _storage_client = None
    _bucket = None

    def __post_init__(self):
        gcp_account_json = self.gcp_account_json
        if gcp_account_json is None:
            gcp_account_json = os.environ.get("GCLOUD_SA_JSON")
            if gcp_account_json is None:
                raise Exception("GCLOUD_SA_JSON environment secret is not set")

        gcp_account_info = json.loads(gcp_account_json)

        from google.cloud.storage import Client

        self._storage_client = Client.from_service_account_info(gcp_account_info)
        self._bucket = self._storage_client.get_bucket(self.bucket_name)

    @property
    def storage_client(self):
        if self._storage_client is None:
            raise Exception("Google Storage client is not initialized")

        return self._storage_client

    @property
    def bucket(self):
        if self._bucket is None:
            raise Exception("Google Storage bucket is not initialized")

        return self._bucket

    @retry(max_retries=3, base_delay=1, backoff_type="exponential", jitter=True)
    def save(self, data: FileData) -> str:
        destination_path = posixpath.join(
            self.folder,
            f"{uuid.uuid4().hex}_{data.file_name}",
        )

        gcp_blob = self.bucket.blob(destination_path)
        gcp_blob.upload_from_string(data.data, content_type=data.content_type)

        if self.url_expiration is None:
            return gcp_blob.public_url

        return gcp_blob.generate_signed_url(
            version="v4",
            expiration=datetime.timedelta(minutes=self.url_expiration),
            method="GET",
        )

from __future__ import annotations

import json
import os
import posixpath
import uuid
from dataclasses import dataclass
from io import BytesIO
from typing import Optional

from fal.toolkit.file.types import FileData, FileRepository
from fal.toolkit.utils.retry import retry

DEFAULT_URL_TIMEOUT = 60 * 15  # 15 minutes


@dataclass
class R2Repository(FileRepository):
    bucket_name: str = "fal_file_storage"
    url_expiration: int = DEFAULT_URL_TIMEOUT
    r2_account_json: str | None = None
    key: str = ""

    _storage_client = None
    _bucket = None

    def __post_init__(self):
        import boto3
        from botocore.client import Config

        r2_account_json = self.r2_account_json
        if r2_account_json is None:
            r2_account_json = os.environ.get("R2_CREDS_JSON")
            if r2_account_json is None:
                raise Exception("R2_CREDS_JSON environment secret is not set")

        r2_account_info = json.loads(r2_account_json)
        account_id = r2_account_info["ACCOUNT_ID"]
        access_key_id = r2_account_info["ACCESS_KEY_ID"]
        secret_access_key = r2_account_info["SECRET_ACCESS_KEY"]

        self._s3_client = boto3.client(
            "s3",
            endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            config=Config(signature_version="s3v4"),
        )
        self._s3_resource = boto3.resource(
            "s3",
            endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
        )

        self._bucket = self._s3_resource.Bucket(self.bucket_name)

    @property
    def storage_client(self):
        if self._s3_resource is None:
            raise Exception("S3 Resource is not initialized")

        return self._s3_resource

    @property
    def bucket(self):
        if self._bucket is None:
            raise Exception("S3 bucket is not initialized")

        return self._bucket

    @retry(max_retries=3, base_delay=1, backoff_type="exponential", jitter=True)
    def save(
        self,
        data: FileData,
        object_lifecycle_preference: Optional[dict[str, str]] = None,
    ) -> str:
        destination_path = posixpath.join(
            self.key,
            f"{uuid.uuid4().hex}_{data.file_name}",
        )

        s3_object = self.bucket.Object(destination_path)
        s3_object.upload_fileobj(
            BytesIO(data.data), ExtraArgs={"ContentType": data.content_type}
        )

        public_url = self._s3_client.generate_presigned_url(
            ClientMethod="get_object",
            Params={"Bucket": self.bucket_name, "Key": destination_path},
            ExpiresIn=self.url_expiration,
        )
        return public_url

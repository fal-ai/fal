from __future__ import annotations

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
class S3Repository(FileRepository):
    bucket_name: str = "fal_file_storage"
    url_expiration: int = DEFAULT_URL_TIMEOUT
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None

    _s3_client = None

    def __post_init__(self):
        try:
            import boto3
            from botocore.client import Config
        except ImportError:
            raise Exception("boto3 is not installed")

        if self.aws_access_key_id is None:
            self.aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID")
            if self.aws_access_key_id is None:
                raise Exception("AWS_ACCESS_KEY_ID environment variable is not set")

        if self.aws_secret_access_key is None:
            self.aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
            if self.aws_secret_access_key is None:
                raise Exception("AWS_SECRET_ACCESS_KEY environment variable is not set")

        self._s3_client = boto3.client(
            "s3",
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            config=Config(signature_version="s3v4"),
        )

    @property
    def storage_client(self):
        if self._s3_client is None:
            raise Exception("S3 client is not initialized")

        return self._s3_client

    @retry(max_retries=3, base_delay=1, backoff_type="exponential", jitter=True)
    def save(
        self,
        data: FileData,
        object_lifecycle_preference: Optional[dict[str, str]] = None,
        key: Optional[str] = None,
    ) -> str:
        destination_path = posixpath.join(
            key or "",
            f"{uuid.uuid4().hex}_{data.file_name}",
        )

        self.storage_client.upload_fileobj(
            BytesIO(data.data),
            self.bucket_name,
            destination_path,
            ExtraArgs={"ContentType": data.content_type},
        )

        public_url = self.storage_client.generate_presigned_url(
            ClientMethod="get_object",
            Params={"Bucket": self.bucket_name, "Key": destination_path},
            ExpiresIn=self.url_expiration,
        )
        return public_url

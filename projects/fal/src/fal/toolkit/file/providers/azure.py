from __future__ import annotations

import os
from dataclasses import dataclass

from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
from fal.toolkit.file.types import FileData, FileRepository


@dataclass
class AzureBlobStorageRepository(FileRepository):
    account_name: str = "fal_ai_account"
    containter_name: str = "artifacts"
    account_key: str | None = None
    folder: str = "artifacts"

    _storage_client = None

    def __post_init__(self):
        account_key = self.account_key
        if account_key is None:
            account_key = os.environ.get("AZURE_STORAGE_ACCOUNT_KEY")
            if account_key is None:
                raise Exception("AZURE_STORAGE_ACCOUNT_KEY environment secret is not set")


        account_url = f"https://{self.account_name}.blob.core.windows.net"
        default_credential = DefaultAzureCredential()
        self.blob_service = BlobServiceClient(account_url, default_credential)
        self.container_client = self.blob_service.get_container_client(self.containter_name)
        self._storage_client = self.container_client
    @property
    def storage_client(self):
        if self._storage_client is None:
            raise Exception("Azure Storage client is not initialized")

        return self._storage_client

    def save(self, data: FileData) -> str:
        destination_path = os.path.join(self.folder, data.file_name)
        container_client = self.blob_service.get_container_client(self.containter_name)
        blob_client = container_client.get_blob_client(self.containter_name, data.file_name)

        with open(file=destination_path, mode="rb") as file_data:
            blob_client.upload_blob(file_data)
        return blob_client.url


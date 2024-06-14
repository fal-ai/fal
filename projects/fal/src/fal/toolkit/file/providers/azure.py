from __future__ import annotations

import datetime
import json
import os
from dataclasses import dataclass

from azure.storage.blob import (
    BlobServiceClient
)
from azure.identity import DefaultAzureCredential
from fal.toolkit.file.types import FileData, FileRepository

@dataclass
class AzureBlobStorageRepository(FileRepository):
    account_name: str = "artifacts"
    containter_name: str = "fal_ai_container"
    account_key: str | None = None
    
    _storage_client = None
    
    def __post_init__(self):
        account_key = self.account_key
        if account_key is None:
            account_key = os.environ.get("AZURE_STORAGE_ACCOUNT_KEY")
            if account_key is None:
                raise Exception("AZURE_STORAGE_ACCOUNT_KEY environment secret is not set")
        
             
        account_url = f"https://{self.account_name}.blob.core.windows.net"  
        default_credential = DefaultAzureCredential() 
        blob_service = BlobServiceClient(account_url, default_credential)
        container_client = blob_service.get_container_client(self.containter_name)
        self._storage_client = container_client
    @property
    def storage_client(self):
        if self._storage_client is None:
            raise Exception("Azure Storage client is not initialized")

        return self._storage_client
    
    def save(self, data: FileData) -> str:
        destination_path = os.path.join(self.folder, data.file_name)
        blob_client = self._storage_client.get_blob_client(self.containter_name, data.file_name)
        
        with open(file=destination_path, mode="rb") as data:
            blob_client.upload_blob(data)
        return blob_client.url
        
        
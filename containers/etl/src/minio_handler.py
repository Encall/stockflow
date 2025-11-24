"""
Handles all MinIO interactions, including downloading and uploading data.
"""
from minio import Minio
from minio.error import S3Error
import os
from pathlib import Path
from typing import List

class MinioHandler:
    """Handles all MinIO interactions."""

    def __init__(self):
        """Initializes the MinIO client."""
        self.client = Minio(
            endpoint=os.getenv("AWS_S3_ENDPOINT_URL"),
            access_key=os.getenv("AWS_ACCESS_KEY_ID"),
            secret_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region=os.getenv("AWS_REGION"),
            secure=True,
        )
        self.bucket_name = "stockflow"

    def download_data(self, local_data_dir: str, prefix: str = "raw/", level_dir: str = "bronze") -> list:
        """Downloads all data from the specified prefix in the bucket."""
        target_dir = os.path.join(local_data_dir, level_dir)
        downloaded_files = []

        try:
            if not self.client.bucket_exists(self.bucket_name):
                print(f"Bucket '{self.bucket_name}' not found.")
                return downloaded_files

            objects = self.client.list_objects(self.bucket_name, prefix=prefix, recursive=True)
            for obj in objects:
                try:
                    filename = os.path.basename(obj.object_name)
                    if not filename:
                        continue

                    local_path = os.path.join(target_dir, filename)
                    os.makedirs(target_dir, exist_ok=True)
                    
                    self.client.fget_object(self.bucket_name, obj.object_name, local_path)
                    downloaded_files.append(filename)
                except S3Error as e:
                    print(f"Error downloading {obj.object_name}: {e}")
        except S3Error as e:
            print(f"Error connecting to MinIO: {e}")

        return downloaded_files

    def upload_data(self, local_data_dir: str, layer: str) -> List[str]:
        """Uploads all files from a specific layer to MinIO."""
        layer_dir = Path(os.path.join(local_data_dir, layer))
        uploaded_files = []

        if not layer_dir.exists():
            print(f"Source directory for layer '{layer}' not found at {layer_dir}")
            return uploaded_files

        for format_dir in layer_dir.iterdir():
            if format_dir.is_dir():
                file_format = format_dir.name
                prefix = f"{layer}/{file_format}/"
                
                for file_path in format_dir.glob("*.*"):
                    try:
                        object_name = prefix + file_path.name
                        self.client.fput_object(self.bucket_name, object_name, str(file_path))
                        uploaded_files.append(object_name)
                    except S3Error as e:
                        print(f"Error uploading {file_path.name}: {e}")
        
        return uploaded_files

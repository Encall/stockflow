"""
Handles all MinIO interactions, including downloading and uploading data.
"""
from minio import Minio
from minio.error import S3Error
import os
from pathlib import Path
from typing import List

def download_data(local_data_dir: str) -> list:
    """Download all data from MinIO raw prefix and return list of filenames"""
    bronze_dir = os.path.join(local_data_dir, "bronze")

    client = Minio(
        endpoint=os.getenv("MINIO_ENDPOINT"),
        access_key=os.getenv("MINIO_ACCESS_KEY"),
        secret_key=os.getenv("MINIO_SECRET_KEY"),
        region=os.getenv("MINIO_REGION"),
        secure=True,
    )

    bucket_name = "stockflow"
    downloaded_files = []

    try:
        buckets = client.list_buckets()
        found_bucket = any(bucket.name == bucket_name for bucket in buckets)

        if found_bucket:
            # Only download from raw/ prefix
            objects = client.list_objects(bucket_name, prefix="raw/", recursive=True)
            for obj in objects:
                try:
                    filename = os.path.basename(obj.object_name)
                    if not filename: continue # Skip directories

                    local_path = os.path.join(bronze_dir, filename)
                    
                    os.makedirs(bronze_dir, exist_ok=True)
                    
                    client.fget_object(bucket_name, obj.object_name, local_path)
                    downloaded_files.append(filename)
                except S3Error as e:
                    print(f"Error downloading {obj.object_name}: {e}")
        else:
            print(f"Bucket '{bucket_name}' not found.")

    except S3Error as e:
        print(f"Error connecting to MinIO or listing buckets: {e}")

    return downloaded_files


def upload_data(local_data_dir: str, layer: str) -> List[str]:
    """
    Upload all files (CSV and Parquet) from a specific data layer's subdirectories to MinIO.

    Args:
        local_data_dir (str): The root data directory path.
        layer (str): The data layer to upload (e.g., "silver", "gold").

    Returns:
        List[str]: A list of uploaded object names.
    """
    layer_dir = Path(os.path.join(local_data_dir, layer))
    bucket_name = "stockflow"
    uploaded_files = []

    client = Minio(
        endpoint=os.getenv("MINIO_ENDPOINT"),
        access_key=os.getenv("MINIO_ACCESS_KEY"),
        secret_key=os.getenv("MINIO_SECRET_KEY"),
        region=os.getenv("MINIO_REGION"),
        secure=True,
    )

    if not layer_dir.exists():
        print(f"Source directory for layer '{layer}' not found at {layer_dir}")
        return uploaded_files

    # Iterate over subdirectories like 'csv' and 'parquet'
    for format_dir in layer_dir.iterdir():
        if format_dir.is_dir():
            file_format = format_dir.name  # 'csv' or 'parquet'
            prefix = f"{layer}/{file_format}/"
            
            files_to_upload = list(format_dir.glob("*.*"))
            
            if not files_to_upload:
                print(f"No files found in {format_dir}")
                continue

            for file_path in files_to_upload:
                try:
                    object_name = prefix + file_path.name
                    
                    client.fput_object(
                        bucket_name,
                        object_name,
                        str(file_path)
                    )
                    
                    uploaded_files.append(object_name)
                    
                except S3Error as e:
                    print(f"Error uploading {file_path.name}: {e}")
    
    return uploaded_files

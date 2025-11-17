from minio import Minio
from minio.error import S3Error
import os
from pathlib import Path
from typing import List

def upload_data(data_dir: str = "../../data/cleaned/", bucket_name: str = "stockflow", prefix: str = "cleaned/") -> List[str]:
    """Upload cleaned data to MinIO bucket with prefix"""
    # ตั้งค่าการเชื่อมต่อ MinIO
    client = Minio(
        "api.minio.encall.space",
        access_key="REMOVED",
        secret_key="REDACTED",
        region="us-east-1"
    )
    
    uploaded_files = []
    
    # หาไฟล์ parquet ทั้งหมดในโฟลเดอร์ cleaned
    path = Path(data_dir)
    parquet_files = list(path.glob("*.parquet"))
    
    if not parquet_files:
        return uploaded_files
    
    for file_path in parquet_files:
        try:
            # object name ใน MinIO จะเป็น "cleaned/filename.parquet"
            object_name = prefix + file_path.name
            
            # Upload ไฟล์
            client.fput_object(
                bucket_name,
                object_name,
                str(file_path)
            )
            
            uploaded_files.append(object_name)
            
        except S3Error as e:
            print(f"Error uploading {file_path.name}: {e}")
    
    return uploaded_files

def upload_single_file(filename: str, data_dir: str = "../../data/cleaned/", bucket_name: str = "stockflow", prefix: str = "cleaned/") -> str:
    """Upload a single file to MinIO"""
    client = Minio(
        "api.minio.encall.space",
        access_key="REMOVED",
        secret_key="REDACTED",
        region="us-east-1"
    )
    
    file_path = os.path.join(data_dir, filename)
    object_name = prefix + filename
    
    try:
        client.fput_object(
            bucket_name,
            object_name,
            file_path
        )
        return object_name
    except S3Error as e:
        raise Exception(f"Failed to upload {filename}: {e}")

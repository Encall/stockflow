from minio import Minio
from minio.error import S3Error
import os


def download_data(data_dir: str = "../../data/") -> list:
    """Download all data from MinIO and return list of filenames"""

    client = Minio(
        endpoint=os.getenv("MINIO_ENDPOINT"),
        access_key=os.getenv("MINIO_ACCESS_KEY"),
        secret_key=os.getenv("MINIO_SECRET_KEY"),
        region=os.getenv("MINIO_REGION"),
        secure=True,
    )

    bucket_name = "stockflow"
    downloaded_files = []

    buckets = client.list_buckets()
    for bucket in buckets:
        if bucket.name == "stockflow":
            objects = client.list_objects(bucket.name, recursive=True)
            for obj in objects:
                try:
                    filename = os.path.basename(obj.object_name)
                    local_path = os.path.join(data_dir, filename)

                    # Create data directory if not exists
                    os.makedirs(data_dir, exist_ok=True)

                    client.fget_object(
                        bucket_name, obj.object_name, local_path)
                    downloaded_files.append(filename)
                except S3Error as e:
                    print(f"Error downloading {obj.object_name}: {e}")

    return downloaded_files

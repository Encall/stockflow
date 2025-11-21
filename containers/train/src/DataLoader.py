from minio import Minio
from minio.error import S3Error
import os
from pathlib import Path
from typing import List
from dotenv import load_dotenv



class DataLoader():
    def __init__(self, stock_name):
        self.stock_name = stock_name

    def _download_data(self, local_data_dir = "./") -> list:
        """Download all data from MinIO raw prefix and return list of filenames"""
        load_dotenv()
        data_dir = os.path.join(local_data_dir, "data")

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
        try:
            buckets = client.list_buckets()
            found_bucket = any(bucket.name == bucket_name for bucket in buckets)

            if found_bucket:
                objects = client.list_objects(bucket_name, prefix="gold/", recursive=True)
                for obj in objects:
                    try:
                        filename = os.path.basename(obj.object_name)
                        if not filename: continue # Skip directories
                        
                        # Filter by stock name and .parquet extension
                        if not filename.startswith(f"{self.stock_name}_") or not filename.endswith(".parquet"):
                            continue

                        local_path = os.path.join(data_dir, filename)
                        
                        os.makedirs(data_dir, exist_ok=True)
                        
                        client.fget_object(bucket_name, obj.object_name, local_path)
                        downloaded_files.append(filename)
                    except S3Error as e:
                        print(f"Error downloading {obj.object_name}: {e}")
            else:
                print(f"Bucket '{bucket_name}' not found.")

        except S3Error as e:
            print(f"Error connecting to MinIO or listing buckets: {e}")

        return downloaded_files
    
    def get_data(self):

        parquet_path = self._download_data()
        if parquet_path:
            print(f"Data downloaded: {parquet_path}")
        else:
            print("No data downloaded.")
            return None
        
        import pandas as pd
        df = pd.read_parquet(os.path.join("./data", parquet_path[0]))
        df = df.replace([float('inf'), float('-inf')], float('nan'))
        return df

if __name__ == "__main__":
    data_loader = DataLoader("DIG")
    f = data_loader.get_data()
    print(f.head())
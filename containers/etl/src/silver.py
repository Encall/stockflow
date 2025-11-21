"""
Functions for processing the Silver data layer.
Includes data cleaning.
"""
import pandas as pd
import os
from pathlib import Path
from typing import List
import argparse
import minio_handler as MinioHandler

class SilverProcessor:
    """Processes the silver data layer."""

    def __init__(self, local_data_dir: str):
        """Initializes the SilverProcessor."""
        self.local_data_dir = local_data_dir
        self.bronze_dir = os.path.join(local_data_dir, "bronze")
        self.silver_csv_dir = os.path.join(local_data_dir, "silver", "csv")
        self.silver_parquet_dir = os.path.join(local_data_dir, "silver", "parquet")
        os.makedirs(self.silver_csv_dir, exist_ok=True)
        os.makedirs(self.silver_parquet_dir, exist_ok=True)

    def _clean_data_logic(self, df: pd.DataFrame, filename: str = "") -> pd.DataFrame:
        """Contains the core logic for cleaning a stock data DataFrame."""
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        df = df.drop_duplicates()
        df = df.dropna()
        df = df[(df['open'] > 0) & (df['high'] > 0) & (df['low'] > 0) & (df['close'] > 0)]
        df = df[df['volume'] >= 0]
        df = df[df['high'] >= df['low']]
        return df

    def process_files(self, file_list: List[str]) -> dict:
        """
        Cleans all specified files from the bronze directory and saves them
        to the silver directory in both CSV and Parquet formats.
        """
        results = {}
        for filename in file_list:
            try:
                bronze_path = os.path.join(self.bronze_dir, filename)
                df = pd.read_parquet(bronze_path)
                df_cleaned = self._clean_data_logic(df, filename)
                
                csv_filename = filename.replace(".parquet", ".csv")
                silver_csv_path = os.path.join(self.silver_csv_dir, csv_filename)
                silver_parquet_path = os.path.join(self.silver_parquet_dir, filename)
                
                df_cleaned.to_csv(silver_csv_path, index=False)
                print(f"Saving cleaned data to {silver_csv_path}")
                df_cleaned.to_parquet(silver_parquet_path, index=False)
                print(f"Saved cleaned data to {silver_parquet_path}")
                
                results[filename] = {"status": "success", "path": silver_csv_path}
            except Exception as e:
                results[filename] = {"status": "failed", "error": str(e)}
        return results

def process_silver_layer(local_data_dir: str, file_list: list[str]) -> dict:
    """
    Orchestrates the silver layer processing by cleaning data from bronze
    and saving it to the silver layer.
    """
    processor = SilverProcessor(local_data_dir)
    return processor.process_files(file_list)


if __name__ == "__main__":
    minio_handler = MinioHandler.MinioHandler()

    parser = argparse.ArgumentParser(description="Process silver layer files.")
    parser.add_argument("--local_data_dir", type=str, required=True, help="Path to local data directory")
    args = parser.parse_args()

    file_names = minio_handler.download_data(local_data_dir=args.local_data_dir, prefix="bronze/", level_dir="bronze")
    print(f"Downloaded files for silver processing")
    process_silver_layer(local_data_dir=args.local_data_dir, file_list=file_names)
    print("Silver layer processing completed.")
    minio_handler.upload_data(local_data_dir=args.local_data_dir, layer="silver")
    print("Silver layer upload completed.")
    
    # /tmp/stockflow/silver/csv/XBI_data.csv

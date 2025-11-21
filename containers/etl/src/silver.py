"""
Functions for processing the Silver data layer.
Includes data cleaning.
"""
import pandas as pd
import os
from pathlib import Path
from typing import List

def _clean_data_logic(df: pd.DataFrame, filename: str = "") -> pd.DataFrame:
    """Contains the core logic for cleaning a stock data DataFrame."""
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # Remove duplicates, missing values, invalid prices, and negative volume
    df = df.drop_duplicates()
    df = df.dropna()
    df = df[(df['open'] > 0) & (df['high'] > 0) & (df['low'] > 0) & (df['close'] > 0)]
    df = df[df['volume'] >= 0]
    df = df[df['high'] >= df['low']]
    
    return df

def _clean_all_files(local_data_dir: str, file_list: list[str]) -> dict:
    """
    Cleans all specified files from the bronze directory and saves them
    to the silver directory in both CSV and Parquet formats.
    """
    BRONZE_DIR = os.path.join(local_data_dir, "bronze")
    SILVER_CSV_DIR = os.path.join(local_data_dir, "silver", "csv")
    SILVER_PARQUET_DIR = os.path.join(local_data_dir, "silver", "parquet")
    
    os.makedirs(SILVER_CSV_DIR, exist_ok=True)
    os.makedirs(SILVER_PARQUET_DIR, exist_ok=True)
    
    results = {}
    
    for filename in file_list:
        try:
            bronze_path = os.path.join(BRONZE_DIR, filename)
            df = pd.read_parquet(bronze_path)
            df_cleaned = _clean_data_logic(df, filename)
            
            csv_filename = filename.replace(".parquet", ".csv")
            silver_csv_path = os.path.join(SILVER_CSV_DIR, csv_filename)
            silver_parquet_path = os.path.join(SILVER_PARQUET_DIR, filename)
            
            df_cleaned.to_csv(silver_csv_path, index=False)
            df_cleaned.to_parquet(silver_parquet_path, index=False)
            
            results[filename] = {"status": "success", "path": silver_csv_path}
        except Exception as e:
            results[filename] = {"status": "failed", "error": str(e)}
            
    return results

def process_silver_layer(local_data_dir: str, file_list: list[str]) -> dict:
    """
    Orchestrates the silver layer processing by cleaning data from bronze
    and saving it to the silver layer.
    """
    clean_results = _clean_all_files(local_data_dir, file_list)
    return clean_results


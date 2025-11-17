import pandas as pd
import os
from pathlib import Path
from typing import List

def load_data(files_name: str, data_dir: str = "../../data/") -> pd.DataFrame:
    """Load parquet file"""
    file_path = data_dir + files_name
    df = pd.read_parquet(file_path)
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean stock data"""
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Sort by date
    df = df.sort_values('date').reset_index(drop=True)
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Remove missing values
    df = df.dropna()
    
    # Remove invalid prices (<=0)
    df = df[(df['open'] > 0) & (df['high'] > 0) & (df['low'] > 0) & (df['close'] > 0)]
    
    # Remove negative volume
    df = df[df['volume'] >= 0]
    
    # Validate price consistency (high >= low, high >= open, high >= close, etc.)
    df = df[(df['high'] >= df['low']) & 
            (df['high'] >= df['open']) & 
            (df['high'] >= df['close']) &
            (df['low'] <= df['open']) &
            (df['low'] <= df['close'])]
    
    return df

def save_data(df: pd.DataFrame, filename: str, output_dir: str = "../../data/cleaned/"):
    """Save cleaned data to parquet"""
    os.makedirs(output_dir, exist_ok=True)
    output_path = output_dir + filename
    df.to_parquet(output_path, index=False)
    return output_path

def get_all_parquet_files(data_dir: str = "../../data/") -> List[str]:
    """Get all parquet files in directory, excluding cleaned subfolder"""
    path = Path(data_dir)
    parquet_files = []
    
    for file in path.glob("*.parquet"):
        parquet_files.append(file.name)
    
    return sorted(parquet_files)

def clean_single_file(filename: str, data_dir: str = "../../data/", output_dir: str = "../../data/cleaned/") -> pd.DataFrame:
    """Clean a single parquet file and save to cleaned directory"""
    # Load
    df = load_data(filename, data_dir)
    
    # Clean
    df_clean = clean_data(df)
    
    # Save (same filename without _cleaned suffix)
    save_data(df_clean, filename, output_dir)
    
    return df_clean

def clean_all_files(data_dir: str = "../../data/", output_dir: str = "../../data/cleaned/", file_list: list = None) -> dict:
    """Clean all parquet files in the data directory or specific files from file_list"""
    if file_list:
        files = [f for f in file_list if f.endswith('.parquet')]
    else:
        files = get_all_parquet_files(data_dir)
    
    results = {}
    for filename in files:
        try:
            df = clean_single_file(filename, data_dir, output_dir)
            results[filename] = {"status": "success", "rows": len(df)}
        except Exception as e:
            results[filename] = {"status": "failed", "error": str(e)}
    
    return results
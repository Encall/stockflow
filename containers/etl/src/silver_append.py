"""
Silver layer append mode for incremental data updates.
Cleans new bronze data and appends to existing silver files.
"""
import pandas as pd
import os
from typing import List
import argparse
import minio_handler as MinioHandler


class SilverAppendProcessor:
    """Handles appending cleaned data to existing silver files."""

    def __init__(self, local_data_dir: str):
        """Initializes the SilverAppendProcessor."""
        self.local_data_dir = local_data_dir
        self.bronze_dir = os.path.join(local_data_dir, "bronze", "parquet")
        self.silver_csv_dir = os.path.join(local_data_dir, "silver", "csv")
        self.silver_parquet_dir = os.path.join(local_data_dir, "silver", "parquet")
        os.makedirs(self.silver_csv_dir, exist_ok=True)
        os.makedirs(self.silver_parquet_dir, exist_ok=True)

    def _clean_data_logic(self, df: pd.DataFrame) -> pd.DataFrame:
        """Contains the core logic for cleaning a stock data DataFrame."""
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        df = df.drop_duplicates()
        df = df.dropna()
        df = df[(df['open'] > 0) & (df['high'] > 0) & (df['low'] > 0) & (df['close'] > 0)]
        df = df[df['volume'] >= 0]
        df = df[df['high'] >= df['low']]
        return df

    def append_file(self, filename: str) -> dict:
        """
        Appends cleaned bronze data to existing silver file.
        
        Args:
            filename: Parquet filename to process
        
        Returns:
            Dictionary with processing result
        """
        try:
            bronze_path = os.path.join(self.bronze_dir, filename)
            
            if not os.path.exists(bronze_path):
                print(f"⚠️ Bronze file not found: {bronze_path}")
                return {
                    "status": "skipped",
                    "message": "Bronze file not found"
                }
            
            # Read and clean bronze data
            df_bronze = pd.read_parquet(bronze_path)
            df_cleaned = self._clean_data_logic(df_bronze)
            
            silver_parquet_path = os.path.join(self.silver_parquet_dir, filename)
            
            # Check if silver file exists
            print(f'Processing file: {silver_parquet_path}')
            if os.path.exists(silver_parquet_path):
                # Read existing silver data
                df_existing = pd.read_parquet(silver_parquet_path)
                
                # Ensure date is datetime
                df_existing['date'] = pd.to_datetime(df_existing['date'])
                df_cleaned['date'] = pd.to_datetime(df_cleaned['date'])
                
                # Get latest date in existing
                latest_existing_date = df_existing['date'].max()
                
                # Filter to only new data
                df_new = df_cleaned[df_cleaned['date'] > latest_existing_date]
                
                if df_new.empty:
                    print(f"⚠️ No new data for {filename}")
                    return {
                        "status": "skipped",
                        "message": "No new data after cleaning",
                        "latest_date": latest_existing_date.strftime('%Y-%m-%d')
                    }
                
                # Append and deduplicate
                df_combined = pd.concat([df_existing, df_new], ignore_index=True)
                df_combined = df_combined.sort_values('date').drop_duplicates(subset=['date'], keep='last')
                
                # Save
                df_combined.to_parquet(silver_parquet_path, index=False)
                
                print(f"✅ Appended {len(df_new)} cleaned rows to {filename} (total: {len(df_combined)})")
                
                return {
                    "status": "appended",
                    "path": silver_parquet_path,
                    "new_rows": len(df_new),
                    "total_rows": len(df_combined)
                }
            else:
                # Create new silver file
                df_cleaned.to_parquet(silver_parquet_path, index=False)
                
                print(f"✅ Created new silver file for {filename} ({len(df_cleaned)} rows)")
                
                return {
                    "status": "created",
                    "path": silver_parquet_path,
                    "rows": len(df_cleaned)
                }
                
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e)
            }

    def process_files(self, file_list: List[str]) -> dict:
        """
        Processes multiple files for appending.
        
        Args:
            file_list: List of filenames to process
        
        Returns:
            Dictionary with results for each file
        """
        results = {}
        for filename in file_list:
            print(f'Processing file: {filename}')
            results[filename] = self.append_file(filename)
        return results


def append_silver_data(local_data_dir: str, file_list: List[str]) -> dict:
    """
    Orchestrates the silver append process.
    
    Args:
        local_data_dir: Local directory path
        file_list: List of files to process
    
    Returns:
        Dictionary with processing results
    """
    processor = SilverAppendProcessor(local_data_dir)
    return processor.process_files(file_list)


if __name__ == "__main__":
    minio_handler = MinioHandler.MinioHandler()

    parser = argparse.ArgumentParser(description="Append cleaned data to silver layer.")
    parser.add_argument("--local_data_dir", type=str, required=True, help="Path to local data directory")
    
    args = parser.parse_args()

    print("=" * 60)
    print("🚀 Starting Silver Append Process")
    print(f"Data directory: {args.local_data_dir}")
    print("=" * 60)

    # Download existing silver files
    print("\n📥 Downloading existing silver files from MinIO...")
    silver_files = minio_handler.download_data(
        local_data_dir=args.local_data_dir,
        prefix="silver/parquet/",
        level_dir="silver/parquet"
    )
    print(f"✅ Downloaded {len(silver_files)} existing silver files")
    
    # Download updated bronze files
    print("\n📥 Downloading bronze files from MinIO...")
    bronze_files = minio_handler.download_data(
        local_data_dir=args.local_data_dir,
        prefix="bronze/parquet/",
        level_dir="bronze/parquet"
    )
    print(f"✅ Downloaded {len(bronze_files)} bronze files")
    
    # Process and append
    print("\n🔄 Cleaning and appending data to silver layer...")
    results = append_silver_data(
        local_data_dir=args.local_data_dir,
        file_list=bronze_files
    )
    print('**' * 20)
    
    # Upload updated silver files
    print("\n📤 Uploading updated silver files to MinIO...")
    uploaded_files = minio_handler.upload_data(
        local_data_dir=args.local_data_dir,
        layer="silver"
    )
    print(f"✅ Uploaded {len(uploaded_files)} files")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Silver Append Summary")
    print("=" * 60)
    
    appended_count = len([r for r in results.values() if r["status"] == "appended"])
    created_count = len([r for r in results.values() if r["status"] == "created"])
    skipped_count = len([r for r in results.values() if r["status"] == "skipped"])
    failed_count = len([r for r in results.values() if r["status"] == "failed"])
    
    print(f"✅ Appended: {appended_count}")
    print(f"🆕 Created: {created_count}")
    print(f"⏭️  Skipped: {skipped_count}")
    print(f"❌ Failed: {failed_count}")
    
    if appended_count > 0:
        print("\nAppended files:")
        for filename, result in results.items():
            if result["status"] == "appended":
                print(f"  ✅ {filename}: +{result['new_rows']} rows (total: {result['total_rows']})")
    
    if failed_count > 0:
        print("\nFailed files:")
        for filename, result in results.items():
            if result["status"] == "failed":
                print(f"  ❌ {filename}: {result['error']}")
    
    print("\n🎉 Silver append process completed!")

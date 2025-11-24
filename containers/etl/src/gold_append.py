"""
Gold layer append mode for incremental feature updates.
Creates features for new silver data and appends to existing gold files.
"""
import pandas as pd
import numpy as np
from typing import List
import os
import argparse
import minio_handler as MinioHandler


class GoldAppendProcessor:
    """Handles appending feature-enriched data to existing gold files."""

    def __init__(self, local_data_dir: str):
        """Initializes the GoldAppendProcessor."""
        self.local_data_dir = local_data_dir
        self.silver_dir = os.path.join(local_data_dir, "silver", "parquet")
        self.gold_parquet_dir = os.path.join(local_data_dir, "gold", "parquet")
        os.makedirs(self.gold_parquet_dir, exist_ok=True)

    def _calculate_rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        """Calculates Relative Strength Index (RSI)."""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_macd(self, series: pd.Series, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> pd.DataFrame:
        """Calculates Moving Average Convergence Divergence (MACD)."""
        fast_ema = series.ewm(span=fast_period, adjust=False).mean()
        slow_ema = series.ewm(span=slow_period, adjust=False).mean()
        macd = fast_ema - slow_ema
        signal_line = macd.ewm(span=signal_period, adjust=False).mean()
        macd_hist = macd - signal_line
        return pd.DataFrame({'macd': macd, 'macd_signal': signal_line, 'macd_hist': macd_hist})

    def _calculate_bollinger_bands(self, series: pd.Series, window: int = 20, num_std: int = 2) -> pd.DataFrame:
        """Calculates Bollinger Bands."""
        rolling_mean = series.rolling(window=window).mean()
        rolling_std = series.rolling(window=window).std()
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)
        return pd.DataFrame({'bb_upper': upper_band, 'bb_lower': lower_band, 'bb_middle': rolling_mean})

    def create_gold_features(self, df: pd.DataFrame, lags: List[int] = [1, 3, 5], sma_window: int = 14, volatility_window: int = 20) -> pd.DataFrame:
        """Creates gold layer features for stock data."""
        df = df.copy()
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').set_index('date')
        
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        df['gap_opening'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)

        for lag in lags:
            df[f'return_lag_{lag}'] = df['log_return'].shift(lag)
            
        df[f'sma_{sma_window}'] = df['close'].rolling(window=sma_window).mean()
        df[f'dist_from_sma'] = (df['close'] - df[f'sma_{sma_window}']) / df[f'sma_{sma_window}']
        df[f'volatility_{volatility_window}'] = df['log_return'].rolling(window=volatility_window).std()
        
        df['rsi'] = self._calculate_rsi(df['close'])
        df = df.join(self._calculate_macd(df['close']))
        df = df.join(self._calculate_bollinger_bands(df['close']))

        df['vol_change'] = df['volume'].pct_change()
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month

        return_cols = ['log_return', 'gap_opening', 'vol_change'] + [f'return_lag_{lag}' for lag in lags]
        for col in return_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)

        rolling_cols = [
            f'sma_{sma_window}', f'dist_from_sma', f'volatility_{volatility_window}', 'rsi',
            'macd', 'macd_signal', 'macd_hist', 'bb_upper', 'bb_lower', 'bb_middle'
        ]
        for col in rolling_cols:
            if col in df.columns:
                df[col] = df[col].bfill()
        
        df = df.fillna(0).reset_index()
        return df

    def append_file(self, filename: str) -> dict:
        """
        Creates features for new silver data and appends to existing gold file.
        
        Args:
            filename: Parquet filename to process
        
        Returns:
            Dictionary with processing result
        """
        try:
            silver_path = os.path.join(self.silver_dir, filename)
            
            print(f'Processing file: {silver_path}')
            if not os.path.exists(silver_path):
                print(f"⚠️ Silver file not found: {silver_path}")
                return {
                    "status": "skipped",
                    "message": "Silver file not found"
                }
            
            gold_parquet_path = os.path.join(self.gold_parquet_dir, filename)
            
            # Check if gold file exists
            if os.path.exists(gold_parquet_path):
                # Read existing gold data
                df_gold_existing = pd.read_parquet(gold_parquet_path)
                df_gold_existing['date'] = pd.to_datetime(df_gold_existing['date'])
                latest_gold_date = df_gold_existing['date'].max()
                
                # Read complete silver data (need historical context for features)
                df_silver = pd.read_parquet(silver_path)
                df_silver['date'] = pd.to_datetime(df_silver['date'])
                
                # Create features for ALL silver data (features need historical context)
                df_gold_full = self.create_gold_features(df_silver)
                df_gold_full['date'] = pd.to_datetime(df_gold_full['date'])
                
                # Extract only the NEW rows (dates after latest gold date)
                df_gold_new = df_gold_full[df_gold_full['date'] > latest_gold_date]
                
                if df_gold_new.empty:
                    print(f"⚠️ No new data for {filename}")
                    return {
                        "status": "skipped",
                        "message": "No new data after feature creation",
                        "latest_date": latest_gold_date.strftime('%Y-%m-%d')
                    }
                
                # Append new rows to existing gold data
                df_combined = pd.concat([df_gold_existing, df_gold_new], ignore_index=True)
                df_combined = df_combined.sort_values('date').drop_duplicates(subset=['date'], keep='last')
                
                # Save
                df_combined.to_parquet(gold_parquet_path, index=False)
                
                print(f"✅ Appended {len(df_gold_new)} feature rows to {filename} (total: {len(df_combined)})")
                
                return {
                    "status": "appended",
                    "path": gold_parquet_path,
                    "new_rows": len(df_gold_new),
                    "total_rows": len(df_combined)
                }
            else:
                # Create new gold file
                df_silver = pd.read_parquet(silver_path)
                df_gold = self.create_gold_features(df_silver)
                df_gold.to_parquet(gold_parquet_path, index=False)
                
                print(f"✅ Created new gold file for {filename} ({len(df_gold)} rows)")
                
                return {
                    "status": "created",
                    "path": gold_parquet_path,
                    "rows": len(df_gold)
                }
                
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e)
            }

    def process_files(self, file_list: List[str]) -> dict:
        """
        Processes multiple files for feature creation and appending.
        
        Args:
            file_list: List of filenames to process
        
        Returns:
            Dictionary with results for each file
        """
        results = {}
        for filename in file_list:
            results[filename] = self.append_file(filename)
        return results


def append_gold_data(local_data_dir: str, file_list: List[str]) -> dict:
    """
    Orchestrates the gold append process.
    
    Args:
        local_data_dir: Local directory path
        file_list: List of files to process
    
    Returns:
        Dictionary with processing results
    """
    processor = GoldAppendProcessor(local_data_dir)
    return processor.process_files(file_list)


if __name__ == "__main__":
    minio_handler = MinioHandler.MinioHandler()

    parser = argparse.ArgumentParser(description="Append features to gold layer.")
    parser.add_argument("--local_data_dir", type=str, required=True, help="Path to local data directory")
    
    args = parser.parse_args()

    print("=" * 60)
    print("🚀 Starting Gold Append Process")
    print(f"Data directory: {args.local_data_dir}")
    print("=" * 60)

    # Download existing gold files
    print("\n📥 Downloading existing gold files from MinIO...")
    gold_files = minio_handler.download_data(
        local_data_dir=args.local_data_dir,
        prefix="gold/parquet/",
        level_dir="gold/parquet"
    )
    print(f"✅ Downloaded {len(gold_files)} existing gold files")
    
    # Download updated silver files
    print("\n📥 Downloading silver files from MinIO...")
    silver_files = minio_handler.download_data(
        local_data_dir=args.local_data_dir,
        prefix="silver/parquet/",
        level_dir="silver/parquet"
    )
    print(f"✅ Downloaded {len(silver_files)} silver files")
    
    # Process and append
    print("\n🔄 Creating features and appending to gold layer...")
    results = append_gold_data(
        local_data_dir=args.local_data_dir,
        file_list=silver_files
    )
    
    # Upload updated gold files
    print("\n📤 Uploading updated gold files to MinIO...")
    uploaded_files = minio_handler.upload_data(
        local_data_dir=args.local_data_dir,
        layer="gold"
    )
    print(f"✅ Uploaded {len(uploaded_files)} files")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Gold Append Summary")
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
    
    print("\n🎉 Gold append process completed!")

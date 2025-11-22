"""
Bronze layer append mode for incremental stock data updates.
Fetches latest data from yfinance and appends to existing bronze files.
"""
import pandas as pd
import os
from typing import List, Optional
import argparse
import minio_handler as MinioHandler
from datetime import datetime, timedelta

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("⚠️ yfinance not installed. API ingestion mode will not be available.")


class BronzeAppendProcessor:
    """Handles appending latest stock data to existing bronze files."""

    def __init__(self, local_data_dir: str):
        """Initializes the BronzeAppendProcessor."""
        self.local_data_dir = local_data_dir
        self.bronze_parquet_dir = os.path.join(local_data_dir, "bronze", "parquet")
        os.makedirs(self.bronze_parquet_dir, exist_ok=True)

    def _fetch_latest_data(self, ticker: str, period: str = "1d") -> pd.DataFrame:
        """
        Fetches the latest stock data from yfinance.
        
        Args:
            ticker: Stock ticker symbol
            period: Period to fetch (default: '1d' for latest day)
        
        Returns:
            DataFrame with latest stock data
        """
        if not YFINANCE_AVAILABLE:
            raise ImportError("yfinance package is not installed.")
        
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period=period)
            
            if df.empty:
                raise ValueError(f"No data returned for ticker {ticker}")
            
            # Reset index and standardize
            df = df.reset_index()
            df.columns = df.columns.str.lower()
            df = df.rename(columns={'datetime': 'date'})
            
            # Add act_symbol column as second column
            df.insert(1, 'act_symbol', ticker)
            
            # Rearrange columns: date, act_symbol, open, high, low, close, volume
            required_columns = ['date', 'act_symbol', 'open', 'high', 'low', 'close', 'volume']
            df = df[required_columns]
            
            # Ensure date is datetime
            df['date'] = pd.to_datetime(df['date'])
            
            return df
            
        except Exception as e:
            raise Exception(f"Error fetching data for {ticker}: {str(e)}")

    def append_ticker_data(self, ticker: str, period: str = "1d") -> dict:
        """
        Fetches latest data and appends to existing bronze file.
        
        Args:
            ticker: Stock ticker symbol
            period: Period to fetch
        
        Returns:
            Dictionary with processing result
        """
        try:
            print(f"📊 Fetching latest data for {ticker}...")
            
            # Fetch latest data
            df_new = self._fetch_latest_data(ticker, period)
            
            filename = f"{ticker}_data.parquet" ## (TODO Error on filename pathfinding)
            bronze_path = os.path.join(self.bronze_parquet_dir, filename)
            
            # Check if file exists
            print(f'Checking if bronze file exists for {ticker} at {bronze_path}...')
            if os.path.exists(bronze_path):
                print(f"📂 Existing bronze file found for {ticker}, appending new data...")
                # Read existing data
                df_existing = pd.read_parquet(bronze_path)
                
                # Ensure date columns are datetime for both
                df_existing['date'] = pd.to_datetime(df_existing['date'])
                df_new['date'] = pd.to_datetime(df_new['date'])
                
                # Remove timezone from df_new if present to match df_existing
                if df_new['date'].dt.tz is not None:
                    df_new['date'] = df_new['date'].dt.tz_localize(None)

                print(f'df_exists head:\n{df_existing.head()}')
                print(f'df_new head:\n{df_new.head()}')
                
                # Get the latest date in existing data
                latest_existing_date = df_existing['date'].max()
                
                # Filter new data to only include dates after the latest existing date
                df_new = df_new[df_new['date'] > latest_existing_date]
                
                if df_new.empty:
                    print(f"⚠️ No new data for {ticker} (latest existing: {latest_existing_date})")
                    return {
                        "status": "skipped",
                        "message": "No new data available",
                        "latest_date": latest_existing_date.strftime('%Y-%m-%d')
                    }
                

                # Append new data
                df_combined = pd.concat([df_existing, df_new], ignore_index=True)
                df_combined = df_combined.sort_values('date').drop_duplicates(subset=['date'], keep='last')
                
                # Save combined data
                df_combined.to_parquet(bronze_path, index=False)
                
                print(f"✅ Appended {len(df_new)} new rows to {ticker} (total: {len(df_combined)})")
                
                return {
                    "status": "appended",
                    "path": bronze_path,
                    "new_rows": len(df_new),
                    "total_rows": len(df_combined),
                    "filename": filename
                }
            else:
                print(f"📂 No existing bronze file found for {ticker}, creating new file...")
                # File doesn't exist, create new
                df_new.to_parquet(bronze_path, index=False)
                
                print(f"✅ Created new file for {ticker} ({len(df_new)} rows)")
                
                return {
                    "status": "created",
                    "path": bronze_path,
                    "rows": len(df_new),
                    "filename": filename
                }
                
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e)
            }

    def process_tickers(self, tickers: List[str], period: str = "1d") -> dict:
        """
        Processes multiple tickers for appending latest data.
        
        Args:
            tickers: List of stock ticker symbols
            period: Period to fetch for each ticker
        
        Returns:
            Dictionary with results for each ticker
        """
        results = {}
        for ticker in tickers:
            results[ticker] = self.append_ticker_data(ticker, period)
        return results


def append_bronze_data(
    local_data_dir: str,
    tickers: List[str],
    period: str = "1d"
) -> dict:
    """
    Orchestrates the bronze append process.
    
    Args:
        local_data_dir: Local directory path
        tickers: List of stock tickers
        period: Period to fetch
    
    Returns:
        Dictionary with processing results
    """
    processor = BronzeAppendProcessor(local_data_dir)
    return processor.process_tickers(tickers, period)


if __name__ == "__main__":
    minio_handler = MinioHandler.MinioHandler()

    parser = argparse.ArgumentParser(description="Append latest stock data to bronze layer.")
    parser.add_argument("--local_data_dir", type=str, required=True, help="Path to local data directory")
    parser.add_argument("--tickers", type=str, help="Comma-separated list of stock tickers")
    parser.add_argument("--period", type=str, default="1d", help="Period to fetch (default: 1d)")
    
    args = parser.parse_args()

    print("=" * 60)
    print("🚀 Starting Bronze Append Process")
    print(f"Data directory: {args.local_data_dir}")
    print("=" * 60)

    # Get tickers
    if not args.tickers:
        tickers_str = os.getenv("STOCK_TICKERS", "AAPL,GOOGL,MSFT")
        tickers = tickers_str.split(",")
    else:
        tickers = args.tickers.split(",")
    
    print(f"\n📊 Tickers to update: {', '.join(tickers)}")
    
    # Download existing bronze_yfinance files from MinIO
    print("\n📥 Downloading existing bronze_yfinance files from MinIO...")
    file_names = minio_handler.download_data(
        local_data_dir=args.local_data_dir,
        prefix="bronze/parquet/",
        level_dir="bronze/parquet"
    )
    print(f"✅ Downloaded {len(file_names)} existing files")
    
    # Append new data
    print("\n🔄 Fetching and appending latest data...")
    results = append_bronze_data(
        local_data_dir=args.local_data_dir,
        tickers=tickers,
        period=args.period
    )
    
    # Upload updated files back to MinIO
    print("\n📤 Uploading updated bronze files to MinIO...")
    uploaded_files = minio_handler.upload_data(
        local_data_dir=args.local_data_dir,
        layer="bronze"
    )
    print(f"✅ Uploaded {len(uploaded_files)} files")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Bronze Append Summary")
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
        print("\nAppended tickers:")
        for ticker, result in results.items():
            if result["status"] == "appended":
                print(f"  ✅ {ticker}: +{result['new_rows']} rows (total: {result['total_rows']})")
    
    if skipped_count > 0:
        print("\nSkipped tickers (no new data):")
        for ticker, result in results.items():
            if result["status"] == "skipped":
                print(f"  ⏭️  {ticker}: {result.get('message', 'No new data')}")
    
    if failed_count > 0:
        print("\nFailed tickers:")
        for ticker, result in results.items():
            if result["status"] == "failed":
                print(f"  ❌ {ticker}: {result['error']}")
    
    print("\n🎉 Bronze append process completed!")

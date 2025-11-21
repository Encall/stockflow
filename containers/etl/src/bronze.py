"""
Functions for processing the Bronze data layer.
Includes two ingestion modes:
1. MinIO Raw -> Bronze: Read from raw folder and ingest into bronze folder
2. yfinance API -> Bronze: Fetch from API and ingest into bronze_yfinance folder
"""
import pandas as pd
import os
from typing import List, Optional
import argparse
import minio_handler as MinioHandler

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("⚠️ yfinance not installed. API ingestion mode will not be available.")


class BronzeProcessor:
    """Processes the bronze data layer with dual ingestion modes."""

    def __init__(self, local_data_dir: str, mode: str = "minio"):
        """
        Initializes the BronzeProcessor.
        
        Args:
            local_data_dir: Local directory path for data storage
            mode: Ingestion mode - either 'minio' (raw->bronze) or 'yfinance' (api->bronze_yfinance)
        """
        self.local_data_dir = local_data_dir
        self.mode = mode
        
        # Setup directories based on mode
        if mode == "minio":
            # Mode 1: Raw data from MinIO -> Bronze
            self.source_dir = os.path.join(local_data_dir, "raw")
            self.bronze_parquet_dir = os.path.join(local_data_dir, "bronze", "parquet")
            self.target_layer = "bronze"
        elif mode == "yfinance":
            # Mode 2: yfinance API -> Bronze_yfinance
            self.source_dir = None  # No local source, fetching from API
            self.bronze_parquet_dir = os.path.join(local_data_dir, "bronze_yfinance", "parquet")
            self.target_layer = "bronze_yfinance"
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'minio' or 'yfinance'")
        
        os.makedirs(self.bronze_parquet_dir, exist_ok=True)

    def _validate_data_logic(self, df: pd.DataFrame, filename: str = "") -> pd.DataFrame:
        """
        Contains the core logic for basic validation of stock data DataFrame.
        Ensures required columns exist and have valid data types.
        """
        required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        
        # Check for required columns
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Standardize date column
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        # Ensure numeric columns are properly typed
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df

    def _fetch_stock_data_from_yfinance(
        self, 
        ticker: str, 
        start_date: Optional[str] = None, 
        end_date: Optional[str] = None,
        period: str = "1d"
    ) -> pd.DataFrame:
        """
        Fetches stock data from yfinance API.
        
        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL', 'GOOGL')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            period: Period to download if dates not specified (default: '1d')
                   Valid periods: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
            interval: Data interval (default: '1d')
                   Valid intervals: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
        
        Returns:
            DataFrame with stock data
        """
        if not YFINANCE_AVAILABLE:
            raise ImportError("yfinance package is not installed. Install with: pip install yfinance")
        
        try:
            stock = yf.Ticker(ticker)
            
            if start_date and end_date:
                print('START_STOP')
                df = stock.history(start=start_date, end=end_date)
            else:
                print('PERIOD')
                df = stock.history(period=period)
            
            if df.empty:
                raise ValueError(f"No data returned for ticker {ticker}")
            
            # Reset index to make date a column
            df = df.reset_index()
            
            # Standardize column names to lowercase
            df.columns = df.columns.str.lower()
            print('-----------------------------------')
            print(f'Returning df for {ticker} with columns: {df.columns.tolist()}')
            
            df = df.rename(columns={'datetime': 'date'})
            
            # Select only the required columns
            required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
            df = df[required_columns]
            # print(f'Returning df for {ticker} with columns: {df.columns.tolist()}')
            
            return df
            
        except Exception as e:
            raise Exception(f"Error fetching data for {ticker}: {str(e)}")

    def process_files_from_minio(self, file_list: List[str]) -> dict:
        """
        Mode 1: Processes files from MinIO raw folder to bronze folder.
        Reads parquet files from raw directory, validates them, and saves to bronze.
        
        Args:
            file_list: List of parquet filenames to process
        
        Returns:
            Dictionary with processing results for each file
        """
        results = {}
        
        for filename in file_list:
            try:
                source_path = os.path.join(self.source_dir, filename)
                
                # Read the raw parquet file
                df = pd.read_parquet(source_path)
                
                # Apply basic validation
                df_validated = self._validate_data_logic(df, filename)
                
                # Save to bronze directory
                bronze_parquet_path = os.path.join(self.bronze_parquet_dir, filename)
                df_validated.to_parquet(bronze_parquet_path, index=False)
                print(f"✅ Processed {filename} -> {bronze_parquet_path}")
                
                results[filename] = {
                    "status": "success", 
                    "path": bronze_parquet_path,
                    "rows": len(df_validated)
                }
                
            except Exception as e:
                results[filename] = {"status": "failed", "error": str(e)}
                print(f"❌ Failed to process {filename}: {str(e)}")
        
        return results

    def process_tickers_from_yfinance(
        self, 
        tickers: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: str = "1d"
    ) -> dict:
        """
        Mode 2: Fetches stock data from yfinance API and saves to bronze_yfinance folder.
        
        Args:
            tickers: List of stock ticker symbols
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            period: Period to download if dates not specified
            interval: Data interval (default: '1d')
                   Valid intervals: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
        
        Returns:
            Dictionary with processing results for each ticker
        """
        results = {}
        
        for ticker in tickers:
            try:
                print(f"📊 Fetching data for {ticker} from yfinance...")
                
                # Fetch data from yfinance API
                df = self._fetch_stock_data_from_yfinance(
                    ticker=ticker,
                    start_date=start_date,
                    end_date=end_date,
                    period=period,
                    interval=interval
                )
                print(df.columns)
                # Apply basic validation
                df_validated = self._validate_data_logic(df, ticker)
                
                # Save as parquet file
                filename = f"{ticker}.parquet"
                bronze_path = os.path.join(self.bronze_parquet_dir, filename)
                df_validated.to_parquet(bronze_path, index=False)
                
                results[ticker] = {
                    "status": "success",
                    "path": bronze_path,
                    "rows": len(df_validated),
                    "filename": filename
                }
                print(f"✅ Successfully saved {ticker} ({len(df_validated)} rows) -> {bronze_path}")
                
            except Exception as e:
                results[ticker] = {
                    "status": "failed",
                    "error": str(e)
                }
                print(f"❌ Failed to process {ticker}: {str(e)}")
        
        return results


def process_bronze_from_minio(local_data_dir: str, file_list: List[str]) -> dict:
    """
    Mode 1 orchestration: Processes raw data from MinIO into bronze layer.
    
    Args:
        local_data_dir: Local directory path for data storage
        file_list: List of parquet filenames from raw folder
    
    Returns:
        Dictionary with processing results
    """
    processor = BronzeProcessor(local_data_dir, mode="minio")
    return processor.process_files_from_minio(file_list)


def process_bronze_from_yfinance(
    local_data_dir: str,
    tickers: List[str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    period: str = "1d"
) -> dict:
    """
    Mode 2 orchestration: Fetches data from yfinance API into bronze_yfinance layer.
    
    Args:
        local_data_dir: Local directory path for data storage
        tickers: List of stock ticker symbols to fetch
        start_date: Start date in 'YYYY-MM-DD' format (optional)
        end_date: End date in 'YYYY-MM-DD' format (optional)
        period: Period to download if dates not specified (default: '1y')
        interval: Data interval (default: '1d')
    
    Returns:
        Dictionary with processing results
    """
    processor = BronzeProcessor(local_data_dir, mode="yfinance")
    return processor.process_tickers_from_yfinance(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        period=period
    )


if __name__ == "__main__":
    minio_handler = MinioHandler.MinioHandler()

    parser = argparse.ArgumentParser(description="Process bronze layer files with dual ingestion modes.")
    parser.add_argument("--local_data_dir", type=str, required=True, help="Path to local data directory")
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=["minio", "yfinance"], 
        default="minio",
        help="Ingestion mode: 'minio' (raw->bronze) or 'yfinance' (api->bronze_yfinance)"
    )
    
    # Arguments for yfinance mode
    parser.add_argument("--tickers", type=str, help="Comma-separated list of stock tickers (for yfinance mode)")
    parser.add_argument("--start_date", type=str, help="Start date YYYY-MM-DD (for yfinance mode)")
    parser.add_argument("--end_date", type=str, help="End date YYYY-MM-DD (for yfinance mode)")
    parser.add_argument("--period", type=str, default="1d", help="Period for yfinance (default: 1d)")
    parser.add_argument("--interval", type=str, default="1d", help="Data interval for yfinance (default: 1d)")
    
    args = parser.parse_args()

    print("=" * 60)
    print(f"🚀 Starting Bronze Layer Processing - Mode: {args.mode.upper()}")
    print(f"Data directory: {args.local_data_dir}")
    print("=" * 60)

    if args.mode == "minio":
        # Mode 1: Process from MinIO raw folder
        print("\n📥 Downloading raw files from MinIO...")
        file_names = minio_handler.download_data(
            local_data_dir=args.local_data_dir, 
            prefix="raw/", 
            level_dir="raw"
        )
        print(f"✅ Downloaded {len(file_names)} files from raw folder")
        
        print("\n🔄 Processing files into bronze layer...")
        results = process_bronze_from_minio(
            local_data_dir=args.local_data_dir, 
            file_list=file_names
        )
        
        success_count = len([r for r in results.values() if r["status"] == "success"])
        print(f"✅ Bronze layer processing completed: {success_count}/{len(file_names)} files")
        
        print("\n📤 Uploading bronze data to MinIO...")
        minio_handler.upload_data(local_data_dir=args.local_data_dir, layer="bronze")
        print("✅ Bronze layer upload completed.")
        
    elif args.mode == "yfinance":
        # Mode 2: Fetch from yfinance API
        if not args.tickers:
            # Use environment variable or default
            tickers_str = os.getenv("STOCK_TICKERS", "AAPL,GOOGL,MSFT")
            tickers = tickers_str.split(",")
        else:
            tickers = args.tickers.split(",")
        
        print(f"\n📊 Fetching data for tickers: {', '.join(tickers)}")
        
        results = process_bronze_from_yfinance(
            local_data_dir=args.local_data_dir,
            tickers=tickers,
            start_date=args.start_date,
            end_date=args.end_date,
            period=args.period
        )
        
        success_count = len([r for r in results.values() if r["status"] == "success"])
        print(f"✅ Bronze_yfinance layer processing completed: {success_count}/{len(tickers)} tickers")
        
        print("\n📤 Uploading bronze_yfinance data to MinIO...")
        minio_handler.upload_data(local_data_dir=args.local_data_dir, layer="bronze_yfinance")
        print("✅ Bronze_yfinance layer upload completed.")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Bronze Layer Summary")
    print("=" * 60)
    success_count = len([r for r in results.values() if r["status"] == "success"])
    failed_count = len([r for r in results.values() if r["status"] == "failed"])
    print(f"✅ Successfully processed: {success_count}")
    print(f"❌ Failed: {failed_count}")
    
    if failed_count > 0:
        print("\nFailed items:")
        for name, result in results.items():
            if result["status"] == "failed":
                print(f"  ❌ {name}: {result['error']}")
    
    print("\n🎉 Bronze layer processing completed!")

import pandas as pd
import numpy as np
from typing import List
import os

class GoldProcessor:
    """Processes the gold data layer, creating features and technical indicators."""

    def __init__(self, local_data_dir: str):
        """Initializes the GoldProcessor."""
        self.local_data_dir = local_data_dir
        self.silver_dir = os.path.join(local_data_dir, "silver", "csv")
        self.gold_csv_dir = os.path.join(local_data_dir, "gold", "csv")
        self.gold_parquet_dir = os.path.join(local_data_dir, "gold", "parquet")
        os.makedirs(self.gold_csv_dir, exist_ok=True)
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

    def process_files(self):
        """
        Reads all CSV files from the silver directory, creates gold features,
        and saves them to the gold directory.
        """
        results = {}
        try:
            file_list = [f for f in os.listdir(self.silver_dir) if f.endswith('.csv')]
        except FileNotFoundError:
            print(f"❌ Silver CSV directory not found at {self.silver_dir}")
            return {"status": "failed", "error": "Silver CSV directory not found"}

        print(f"Found {len(file_list)} files in silver directory.")

        for filename in file_list:
            try:
                silver_path = os.path.join(self.silver_dir, filename)
                df_silver = pd.read_csv(silver_path)
                df_gold = self.create_gold_features(df_silver)
                
                parquet_filename = filename.replace(".csv", ".parquet")
                gold_csv_path = os.path.join(self.gold_csv_dir, filename)
                gold_parquet_path = os.path.join(self.gold_parquet_dir, parquet_filename)

                df_gold.to_csv(gold_csv_path, index=False)
                df_gold.to_parquet(gold_parquet_path, index=False)
                
                results[filename] = {"status": "success", "path": gold_csv_path}
            except Exception as e:
                results[filename] = {"status": "failed", "error": str(e)}
        return results

def create_all_gold_files(local_data_dir: str):
    """
    Orchestrates the gold layer processing.
    """
    processor = GoldProcessor(local_data_dir)
    return processor.process_files()


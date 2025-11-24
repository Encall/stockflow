from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Sequence

import pandas as pd
from pandas.api.types import is_numeric_dtype

from minio_handler import MinioClient


@dataclass
class DriftWindow:
    """Container for a single reference/current pair built from a rolling window."""

    reference: pd.DataFrame
    current: pd.DataFrame
    reference_period: tuple[pd.Timestamp, pd.Timestamp]
    current_period: tuple[pd.Timestamp, pd.Timestamp]
    window_index: int


class RollingWindowDataLoader:
    """
    Load gold-layer parquet data (optionally all assets) and yield rolling 60-day windows
    split into 30-day reference/current segments for drift comparison.

    Scaling: only the configured monitoring features are scaled; targets are left untouched.
    """

    def __init__(
        self,
        stock_symbol: str | None = None,
        feature_cols: Optional[Sequence[str]] = None,
        target_col: str | Sequence[str] | None = "close",
        scaler: Optional[object] = None,
        date_col: str = "date",
        window_size: int = 60,
        split_size: int = 30,
        local_cache_dir: str | Path = "data/gold",
    ) -> None:
        if window_size != split_size * 2:
            raise ValueError("window_size must be exactly twice split_size (e.g., 60 and 30).")

        self.stock_symbol = stock_symbol
        self.feature_cols = list(feature_cols) if feature_cols is not None else None
        self.target_col = target_col[0] if isinstance(target_col, Sequence) and not isinstance(target_col, str) else target_col
        self.scaler = scaler
        self.x_scaler: Optional[object] = None
        self.y_scaler: Optional[object] = None
        self.date_col = date_col
        self.window_size = window_size
        self.split_size = split_size
        self.local_cache_dir = Path(local_cache_dir)
        self._client = MinioClient()

    def _resolve_objects(self) -> list[str]:
        """Find all gold/parquet objects matching the configured symbol (or all symbols)."""
        prefix = "gold/parquet/"
        matches: list[str] = []
        for obj in self._client.client.list_objects(self._client.bucket_name, prefix=prefix, recursive=True):
            filename = Path(obj.object_name).name
            if not filename.endswith(".parquet"):
                continue
            if self.stock_symbol and not filename.startswith(f"{self.stock_symbol}_"):
                continue
            matches.append(obj.object_name)

        if not matches:
            target = f" for symbol '{self.stock_symbol}'" if self.stock_symbol else ""
            raise FileNotFoundError(f"No gold parquet objects found{target}.")
        return matches

    def _download_all(self) -> list[Path]:
        """Download matching gold parquet files to the local cache directory."""
        object_names = self._resolve_objects()
        target_dir = self.local_cache_dir / "parquet"
        return [self._client.download(obj_name, target_dir) for obj_name in object_names]

    def _select_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Keep only configured monitoring features (plus date for window metadata)."""
        if self.feature_cols is None:
            cols = [c for c in df.columns if c != self.date_col]
        else:
            cols = list(self.feature_cols)

        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"Columns not found in dataframe: {', '.join(missing)}")

        # Keep the date column at the end for window splitting metadata.
        return df[cols + ([self.date_col] if self.date_col in df.columns else [])].copy()

    def _apply_scaling(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scale only the configured monitoring features using the provided scaler."""
        if self.scaler is None:
            return df

        df = df.copy()
        features = list(self.feature_cols or [])
        if not features:
            return df  # No explicit monitoring features set; skip scaling

        non_numeric = [c for c in features if c and not is_numeric_dtype(df[c])]
        if non_numeric:
            raise ValueError(f"Non-numeric columns cannot be scaled: {', '.join(non_numeric)}")

        self.x_scaler = self.scaler
        X = df[features].values
        df.loc[:, features] = self.x_scaler.fit_transform(X)
        return df

    def load(self) -> pd.DataFrame:
        """Download, combine, clean, and optionally scale all matching gold parquet files."""
        local_paths = self._download_all()

        frames = [pd.read_parquet(path) for path in local_paths]
        df = pd.concat(frames, ignore_index=True, sort=False)
        df = df.replace([float("inf"), float("-inf")], pd.NA)

        if self.date_col in df.columns:
            df[self.date_col] = pd.to_datetime(df[self.date_col], utc=True, errors="coerce")
            # Normalize all timestamps to naive after coercing; drop rows that failed conversion
            df = df.dropna(subset=[self.date_col])
            df[self.date_col] = df[self.date_col].dt.tz_convert(None)
            df = df.sort_values(self.date_col)
            subset_cols = [self.date_col]
            if "act_symbol" in df.columns:
                subset_cols.insert(0, "act_symbol")
            df = df.drop_duplicates(subset=subset_cols, keep="last")

        df = df.dropna().reset_index(drop=True)
        df = self._select_columns(df)
        return self._apply_scaling(df)

    def iter_windows(self, df: pd.DataFrame | None = None) -> Iterator[DriftWindow]:
        """
        Yield rolling windows starting from the most recent date:
        - reference: most recent 30 days
        - current: the previous 30 days (days 31-60 back)
        """
        source_df = self.load() if df is None else df.copy()

        if self.date_col in source_df.columns:
            # Sort newest to oldest so window 0 uses the latest dates
            source_df = source_df.sort_values(self.date_col, ascending=False).reset_index(drop=True)

        if len(source_df) < self.window_size:
            raise ValueError(
                f"Not enough rows to build a {self.window_size}-day window; "
                f"found {len(source_df)} rows."
            )

        # Drop the date column from drift inputs but keep date ranges as metadata.
        has_date = self.date_col in source_df.columns

        for idx, start in enumerate(range(0, len(source_df) - self.window_size + 1, self.split_size)):
            window = source_df.iloc[start : start + self.window_size]

            reference_slice = window.iloc[: self.split_size]
            current_slice = window.iloc[self.split_size : self.window_size]

            if has_date:
                ref_period = (
                    reference_slice[self.date_col].min(),
                    reference_slice[self.date_col].max(),
                )
                cur_period = (
                    current_slice[self.date_col].min(),
                    current_slice[self.date_col].max(),
                )
            else:
                ref_period = cur_period = (pd.NaT, pd.NaT)

            yield DriftWindow(
                reference=reference_slice.drop(columns=[self.date_col], errors="ignore"),
                current=current_slice.drop(columns=[self.date_col], errors="ignore"),
                reference_period=ref_period,
                current_period=cur_period,
                window_index=idx,
            )


if __name__ == "__main__":
    # Example usage:
    # loader = RollingWindowDataLoader(stock_symbol="DIG", feature_cols=["open", "high", "low", "volume"])
    # for window in loader.iter_windows():
    #     print(
    #         f"Window {window.window_index}: "
    #         f"ref {window.reference_period[0].date()}→{window.reference_period[1].date()}, "
    #         f"cur {window.current_period[0].date()}→{window.current_period[1].date()}, "
    #         f"rows {len(window.reference)} / {len(window.current)}"
    #     )
    pass
